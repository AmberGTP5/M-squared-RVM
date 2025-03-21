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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM  # 导入自定义的模型结构


# 查找助手回复的token位置的辅助函数
def find_assistant_tokens(tokenizer, target):
    """
    在输入序列中找出所有助手回复的起始和结束位置
    参数:
    - tokenizer: 分词器
    - target: 输入序列的token ID列表
    返回:
    - result: 包含所有(start, end)位置对的列表
    """
    result = []
    start_index = 0
    end_index = 0
    # 遍历整个序列寻找助手回复部分
    while start_index <= len(target)-1:
        # 如果当前token不是"assistant"标记，继续向前
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            # 找到"assistant"标记后，向前寻找结束标记
            end_index += 1
            # 当找到结束标记时，记录这对位置
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index+1, end_index+1))
                start_index = end_index+1
    return result

# 定义SFT数据集类 - 适配Geo170K格式
class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        """
        初始化SFT数据集
        参数:
        - images_path: 图像文件夹路径
        - data_path: 数据JSON文件路径
        - tokenizer: 分词器
        - processor: 图像处理器
        - config: 模型配置
        """
        super().__init__()
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # 加载Geo170K格式的数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        print(f"加载了{len(self.data)}个训练样本")
        
        # 验证几个样本的图像路径
        import random
        samples = random.sample(self.data, min(5, len(self.data)))
        print("\n图像路径检查:")
        for i, item in enumerate(samples):
            img_path = os.path.join(self.images_path, item['image'])
            print(f"样本 {i+1}: {img_path} - {'存在' if os.path.exists(img_path) else '不存在'}")
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.data)
    
    def __getitem__(self, index):
        """获取单个数据样本"""
        try:
            # 获取样本
            sample = self.data[index]
            image_path = sample['image']
            conversations = sample['conversations']
            
            # 提取问题和回答
            human_msg = next(conv['value'] for conv in conversations if conv['from'] == 'human')
            assistant_msg = next(conv['value'] for conv in conversations if conv['from'] in ['gpt', 'assistant'])
            
            # 构建完整图像路径
            full_image_path = os.path.join(self.images_path, image_path)
            
            # 构建对话消息
            messages = [
                {"role": "system", "content": "你是一个几何专家，能够详细解答几何问题。"},
                {"role": "user", "content": human_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(messages, 
                tokenize=False).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            
            # 将文本转换为token ID
            input_ids = self.tokenizer(text)['input_ids']
            
            # 找出助手回复的位置
            indexs = find_assistant_tokens(self.tokenizer, input_ids)
            
            # 创建标签
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            
            # 调整input_ids和labels
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
            # 加载并处理图像
            image = Image.open(full_image_path).convert('RGB')
            pixel_values = self.processor(text=None, images=image)['pixel_values']
            
        except Exception as e:
            print(f"处理第{index}个样本时出错: {e}")
            # 创建默认白色图像
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            
            # 创建默认问题"图片内容是什么"
            q_text = self.tokenizer.apply_chat_template([
                {"role":"system", "content":'你是一个几何专家，能够详细解答几何问题。'}, 
                {"role":"user", "content":"这个几何图像无法加载，请提供一个一般性解答\n<image>"}
            ], tokenize=False).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            
            # 创建默认回答
            a_text = '抱歉，我无法查看这个几何图像。请提供几何问题的详细描述，包括已知条件和需要求解的内容，我将尽力帮助您解答。' + self.tokenizer.eos_token
            
            # 将文本转换为token ID
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            
            # 创建标签
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            
            # 调整input_ids和labels
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
        """
        初始化数据收集器
        参数:
        - tokenizer: 分词器
        """
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
            # 对input_ids和labels进行填充，使其达到批中的最大长度
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        # 返回批处理后的数据，转换为张量
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


# 主函数：模型训练流程
if __name__ == '__main__':
    # 创建模型配置
    config = VLMConfig()
    
    # 加载视觉处理器和文本分词器
    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/multimodal/models/siglip-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/multimodal/models/Qwen2.5-0.5B-Instruct')
    
    # 注册自定义模型配置和模型类
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    
    # 加载预训练好的模型
    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/multimodal/train_multimodal_from_scratch/save/pretrain')
    
    # 设置各部分参数的可训练状态
    # 冻结线性层和视觉模型参数
    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model':
            param.requires_grad = False
        # 只微调语言模型部分
        if 'llm_model' in name:
            param.requires_grad = True
    
    # 打印模型参数量和可训练参数量
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}') 
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}') 
    
    # 设置Geo170K数据路径
    images_path = '/root/autodl-tmp/multimodal/dataset/Geo170K/images'
    data_path = '/root/autodl-tmp/multimodal/dataset/Geo170K/qa_tuning.json'
    output_dir = 'save/geo_sft' 
    
    # 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir,  # 输出目录
        do_train=True,  # 执行训练
        per_device_train_batch_size=4,  # 较小的批大小以适应几何问题
        learning_rate=1e-5,  # 较小的学习率以获得更精细的调整
        num_train_epochs=3,  # 训练轮数
        save_steps=2000,  # 每2000步保存一次
        save_total_limit=2,  # 最多保存2个检查点
        fp16=True,  # 使用混合精度训练
        gradient_accumulation_steps=8,  # 梯度累积步数
        logging_steps=100,  # 每100步记录日志
        report_to='tensorboard',  # 报告到tensorboard
        dataloader_pin_memory=True,  # 使用内存锁定加速数据加载
        dataloader_num_workers=4  # 数据加载器的工作线程数
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SFTDataset(images_path, data_path, tokenizer, processor, config),  # 使用修改后的数据集
        data_collator=MyDataCollator(tokenizer)  # 数据收集器
    )
    
    # 开始训练
    trainer.train(resume_from_checkpoint=False)
    
    # 保存模型和训练状态
    trainer.save_model('save/geo_sft')
    trainer.save_state()