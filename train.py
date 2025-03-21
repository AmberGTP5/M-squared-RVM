# 导入必要的库
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image  # 用于处理图像
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast  # 导入模型输出类型
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any
import torch
import gc

# 清空 PyTorch 缓存
torch.cuda.empty_cache()

# 强制垃圾回收
gc.collect()

# 定义视觉语言模型(VLM)的配置类，继承自Hugging Face的PretrainedConfig
class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"  # 模型类型标识
    def __init__(self,llm_model_path = '/root/autodl-tmp/multimodal/models/Qwen2.5-0.5B-Instruct',  # 语言模型路径
                 vision_model_path = '/root/autodl-tmp/multimodal/models/siglip-base-patch16-224',  # 视觉模型路径
                 freeze_vision_model = True,  # 是否冻结视觉模型参数
                 image_pad_num = 243,  # 图像填充标记数量
                **kwargs):
        # 保存配置参数
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)  # 调用父类初始化方法
        
        
# 定义视觉语言模型类，继承自Hugging Face的PreTrainedModel
class VLM(PreTrainedModel):
    config_class = VLMConfig  # 指定配置类
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 加载视觉模型
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        # 加载语言模型
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        
        # 创建连接视觉和语言模型的线性层
        # 将视觉特征维度转换为语言模型的隐藏维度
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*3, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        
        # 根据配置决定是否冻结视觉模型参数
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # 冻结语言模型参数
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # 获取文本嵌入
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        # 获取图像嵌入
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 
        b, s, d = image_embeds.shape

        # 打印形状以诊断问题
        # print(f"图像特征形状: {image_embeds.shape}")

        # 重塑图像特征，压缩图像token数量 (b, 196, d) --> (b, 49, d*4)
        image_embeds = image_embeds.view(b, -1, d*3)
        # 通过线性层转换图像特征到语言模型的维度空间
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        # 确保文本嵌入和图像特征数据类型一致
        text_embeds = text_embeds.to(image_features.dtype)
        
        # 将图像特征合并到文本嵌入中
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        # 通过语言模型前向传播
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)  # 忽略填充标记
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        """将图像特征合并到输入嵌入中，替换<|image_pad|>标记"""
        num_images, num_image_patches, embed_dim = image_features.shape
        # 找出<|image_pad|>标记的位置
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        # 用图像特征替换这些位置的嵌入
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        
        return inputs_embeds
    
# 定义自定义数据集类
class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path  # 数据JSON文件路径
        self.images_path = images_path  # 图像文件夹路径
        self.tokenizer = tokenizer  # 分词器
        self.processor = processor  # 图像处理器
        self.config = config  # 模型配置
        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.datas)
    
    def __getitem__(self, index):
        """获取单个数据样本"""
        sample = self.datas[index]
        try:
            # 获取图像名称和对话内容
            image_name = sample['image']
            conversations = sample['conversations']
            
            # 处理问题文本，应用聊天模板，并将<image>替换为图像填充标记
            q_text = self.tokenizer.apply_chat_template([
                {"role":"system", "content":'You are a helpful assistant.'}, 
                {"role":"user", "content":conversations[0]['value']}
            ], tokenize=False, add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            
            # 处理回答文本
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            
            # 将文本转换为token ID
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            
            # 合并问题和回答的token ID
            input_ids = q_input_ids + a_input_ids
            
            # 创建标签，问题部分用pad_token_id填充，回答部分使用实际token ID
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            
            # 调整input_ids和labels，使二者错位一位（用于自回归训练）
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
            # 加载并处理图像
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
            
        except:
            # 异常处理：创建默认白色图像
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            
            # 创建默认问答对
            q_text = self.tokenizer.apply_chat_template([
                {"role":"system", "content":'You are a helpful assistant.'}, 
                {"role":"user", "content":"图片内容是什么\n<image>"}
            ], tokenize=False, add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            
            a_text = '图片内容为空' + self.tokenizer.eos_token
            
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        # 返回处理好的样本
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 
     

# 定义数据收集器类，用于批处理
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将一批样本整合为模型输入格式"""
        # 找出最大序列长度
        max_len = max(len(feature['input_ids']) for feature in features)
        
        input_ids = []
        labels = []
        pixel_values = []
        
        # 处理每个样本
        for feature in features:
            # 对input_ids和labels进行填充
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        # 返回批处理后的数据，转换为张量
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }
            
        
# 主函数：模型训练流程        
if __name__ == '__main__':
    # 创建模型配置和模型实例
    config = VLMConfig(vision_model_path='/root/autodl-tmp/multimodal/models/siglip-base-patch16-224', image_pad_num=243)
    model = VLM(config).cuda()  # 将模型移至GPU
    
    # 打印模型结构和可训练参数数量
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # 设置数据路径
    images_path = '/root/autodl-tmp/multimodal/dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = '/root/autodl-tmp/multimodal/dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    
    # 加载分词器和处理器
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    
    # 设置输出目录
    output_dir = 'save/pretrain' 
    
    # 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir,  # 输出目录
        do_train=True,  # 执行训练
        per_device_train_batch_size=16,  # 每个设备的批大小
        learning_rate=5e-4,  # 学习率
        num_train_epochs=5,  # 训练轮数
        save_steps=5000,  # 每500步保存一次
        save_total_limit=1,  # 最多保存2个检查点
        fp16=True,  # 使用混合精度训练
        gradient_accumulation_steps=8,  # 梯度累积步数
        logging_steps=500,  # 每100步记录日志
        report_to='tensorboard',  # 报告到tensorboard
        dataloader_pin_memory=True,  # 使用内存锁定加速数据加载
        dataloader_num_workers=8  # 数据加载器的工作线程数
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),  # 训练数据集
        data_collator=MyDataCollator(tokenizer)  # 数据收集器
    )
    
    # 开始训练
    trainer.train(resume_from_checkpoint=False)
    
    # 保存模型和训练状态
    trainer.save_model('save/pretrain')
    trainer.save_state()