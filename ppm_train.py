# 导入必要的库
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import re
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from train import VLMConfig, VLM  # 导入基础模型结构
import gc
import tqdm  # 添加tqdm导入
import logging  # 也需要添加logging以使用logger

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

# 设置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 清空 PyTorch 缓存
torch.cuda.empty_cache()
gc.collect()


# 定义PPM配置类，扩展VLM配置
class PPMConfig(VLMConfig):
    """Process Reward Model配置,支持验证功能"""
    model_type = "ppm_model"
    
    def __init__(self,
                 llm_model_path='/root/autodl-tmp/multimodal/models/Qwen2.5-0.5B-Instruct',
                 vision_model_path='/root/autodl-tmp/multimodal/models/siglip-base-patch16-224',
                 freeze_vision_model=True,  # 冻结视觉模型
                 freeze_mlp_projections=True,  # 冻结MLP投影层
                 freeze_llm_model=False,  # 不冻结语言模型
                 image_pad_num=243,
                 enable_verification=True,  # 启用验证功能
                 **kwargs):
        super().__init__(llm_model_path, vision_model_path, freeze_vision_model, image_pad_num, **kwargs)
        self.freeze_mlp_projections = freeze_mlp_projections
        self.freeze_llm_model = freeze_llm_model
        self.enable_verification = enable_verification


# 二元预测头：对推理步骤进行正确/错误判断
class BinaryPredictionHead(nn.Module):
    """二元分类头，用于验证推理步骤的正确性"""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)  # 二分类：0=错误，1=正确
        
    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layernorm(x)
        return self.classifier(x)


# 扩展VLM模型，添加验证功能
class PPMModel(VLM):
    """Process Reward Model，能够验证推理步骤的正确性"""
    config_class = PPMConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # 添加二元预测头
        self.verification_head = BinaryPredictionHead(self.llm_model.config.hidden_size)
        
        # 根据配置冻结组件
        if config.freeze_vision_model:
            self._freeze_vision_tower()
            
        if config.freeze_mlp_projections:
            self._freeze_mlp_projection()
            
        if config.freeze_llm_model:
            self._freeze_llm_components()
        else:
            self._unfreeze_llm_components()
    
    def _freeze_vision_tower(self):
        """冻结视觉模型参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def _freeze_mlp_projection(self):
        """冻结MLP投影层参数"""
        for param in self.linear1.parameters():
            param.requires_grad = False
        for param in self.linear2.parameters():
            param.requires_grad = False
    
    def _freeze_llm_components(self):
        """冻结语言模型参数"""
        for param in self.llm_model.parameters():
            param.requires_grad = False
    
    def _unfreeze_llm_components(self):
        """解冻语言模型参数"""
        for param in self.llm_model.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, labels=None, pixel_values=None, attention_mask=None, 
            step_indices=None, verification_labels=None):
        """前向传播，支持验证功能"""
        # 添加防御性检查，确保step_indices不是None
        if step_indices is None:
            step_indices = [[] for _ in range(input_ids.shape[0])]
                
        batch_size = input_ids.shape[0]
        
        # 基本VLM前向传播
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        # 获取图像嵌入
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeds.shape
        
        # 对图像嵌入进行全局平均池化和最大池化，并连接
        avg_pooled = torch.mean(image_embeds, dim=1)  # 全局平均池化
        max_pooled = torch.max(image_embeds, dim=1)[0]  # 全局最大池化
        cls_token = image_embeds[:, 0]  # CLS token
        
        # 拼接不同的池化特征
        pooled_image_embeds = torch.cat([cls_token, avg_pooled, max_pooled], dim=-1)
        
        # 通过两个线性层处理视觉特征
        projection = self.linear1(pooled_image_embeds)
        projection = F.relu(projection)
        projection = self.linear2(projection)
        
        # 在文本嵌入中定位图像填充标记位置并替换为投影特征
        img_pad_start_indices = []
        
        for i in range(batch_size):
            # 在文本中查找图像填充标记
            img_pad_flag = (input_ids[i] == self.tokenizer.convert_tokens_to_ids('<|image_pad|>'))
            start_idx = torch.where(img_pad_flag)[0]
            
            if len(start_idx) > 0:
                img_pad_start_indices.append(start_idx[0].item())
            else:
                img_pad_start_indices.append(-1)
        
        # 将图像特征插入到图像填充标记位置
        for i in range(batch_size):
            if img_pad_start_indices[i] != -1:
                text_embeds[i, img_pad_start_indices[i]] = projection[i]
        
        # 使用语言模型进行前向计算
        outputs = self.llm_model(
            inputs_embeds=text_embeds,
            labels=labels,
            output_hidden_states=True
        )
        
        loss = outputs.loss
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # 获取最后一层隐藏状态
        
        # 验证损失计算（如果启用验证并提供了步骤索引）
        verification_loss = None
        verification_logits = []
        
        if self.config.enable_verification and step_indices is not None:
            # 收集所有步骤的特征和标签
            batch_verification_logits = []
            batch_verification_labels = []
            
            for b in range(batch_size):
                # 获取当前批次样本的步骤索引
                sample_indices = step_indices[b]
                
                if len(sample_indices) > 0:
                    # 获取步骤位置的隐藏状态
                    step_features = hidden_states[b, sample_indices]
                    
                    # 通过验证头获取预测
                    sample_verification_logits = self.verification_head(step_features)
                    batch_verification_logits.append(sample_verification_logits)
                    
                    # 如果提供了验证标签，收集它们
                    if verification_labels is not None and b < len(verification_labels) and len(verification_labels[b]) > 0:
                        batch_verification_labels.append(verification_labels[b])
            
            # 如果有收集到的验证逻辑和标签，计算损失
            if len(batch_verification_logits) > 0 and verification_labels is not None and len(batch_verification_labels) > 0:
                # 拼接所有步骤的预测和标签
                all_verification_logits = torch.cat(batch_verification_logits, dim=0)
                all_verification_labels = torch.cat(batch_verification_labels, dim=0)
                    
                # 计算二元交叉熵损失，对应论文公式(8)
                verification_loss = F.cross_entropy(all_verification_logits, all_verification_labels)
                    
                # 组合生成损失和验证损失
                if loss is not None:
                    total_loss = loss + verification_loss
                else:
                    total_loss = verification_loss
                    
                loss = total_loss
        
        # 返回结果，包含生成和验证的信息
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 定位推理步骤的函数
def locate_reasoning_steps(instruction, tokenizer):
    """
    定位带有"и"标记的推理步骤并提取验证标签
    
    参数:
    - instruction: 包含推理步骤和验证标记的指令文本
    - tokenizer: 分词器
    
    返回:
    - step_indices: 推理步骤的位置索引
    - verification_labels: 对应的验证标签（0=错误，1=正确）
    """
    step_indices = []
    verification_labels = []
    
    # 如果instruction为空或None，提前返回空列表
    if not instruction:
        return step_indices, verification_labels
    
    # 查找所有带и标记的部分
    pattern = r"(Step \d+:|步骤 ?\d+[：:]).*?и\s*(<pos>|<neg>)?"
    matches = re.finditer(pattern, instruction, re.DOTALL)
    
    for match in matches:
        step_header = match.group(1)  # 例如 "Step 1:" 或 "步骤1："
        step_content = match.group(0)  # 完整匹配内容
        
        # 找到步骤标题的位置
        step_start = instruction.find(step_header)
        if step_start != -1:
            # 获取步骤开始的token索引
            tokens_before = tokenizer.encode(instruction[:step_start], add_special_tokens=False)
            step_token_idx = len(tokens_before)
            step_indices.append(step_token_idx)
            
            # 判断验证标签
            if "<pos>" in step_content:
                verification_labels.append(1)  # 正确
            elif "<neg>" in step_content:
                verification_labels.append(0)  # 错误
            else:
                # 默认为正确
                verification_labels.append(1)
    
    return step_indices, verification_labels


# 定义处理DualMath-1.1M数据集的类
class ProcessRewardDataset(Dataset):
    # 需要修改ProcessRewardDataset类的初始化方法:

    def __init__(self, mathv360k_base_path, dualmathv1_path, tokenizer, processor, config, max_samples=None):
        """初始化数据集"""
        super().__init__()
        self.mathv360k_base_path = mathv360k_base_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # 加载DualMath-1.1M数据 (JSONL格式)
        self.data = []
        with open(dualmathv1_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSONL行: {line[:100]}...")
        
        # 限制样本数（可选）
        if max_samples and max_samples < len(self.data):
            self.data = self.data[:max_samples]
        
        # 过滤有效样本
        self.valid_samples = []
        invalid_count = 0
        
        for sample in tqdm.tqdm(self.data, desc="过滤有效样本"):
            image_url = sample.get('image_url', '')
            if not image_url:
                invalid_count += 1
                continue
                
            # 解析图像路径
            full_path = self._resolve_image_path(image_url)
            
            # 检查图像是否存在
            if os.path.exists(full_path):
                sample['full_image_path'] = full_path
                self.valid_samples.append(sample)
            else:
                invalid_count += 1
        
        logger.info(f"总样本数: {len(self.data)}")
        logger.info(f"有效样本数: {len(self.valid_samples)}")
        logger.info(f"无效样本数: {invalid_count}")
    
    def _resolve_image_path(self, image_url):
        """解析图像路径，处理不同格式"""
        # 处理DualMath-1.1M中的路径
        if image_url.startswith("MathV-360k/"):
            # 移除前缀"MathV-360k/"并添加"data_images/"
            relative_path = image_url[len("MathV-360k/"):]
            return os.path.join(self.mathv360k_base_path, "data_images", relative_path)
        
        # 处理简单路径
        elif not image_url.startswith("data_images/"):
            # 添加"data_images/"前缀
            return os.path.join(self.mathv360k_base_path, "data_images", image_url)
        
        # 已经包含完整路径
        else:
            return os.path.join(self.mathv360k_base_path, image_url)

    def __len__(self):
        # 修改为返回有效样本的数量，而不是原始数据的数量
        return len(self.valid_samples)
    
    def __getitem__(self, index):
        """处理单个数据样本"""
        sample = self.valid_samples[index]
        
        try:
            # 获取图像路径
            full_image_path = sample['full_image_path']
            
            # 获取指令文本 (包含推理步骤和验证标签)
            instruction = sample.get('instruction', '')
            
            # 构建问题文本
            q_text = instruction
            if self.tokenizer.chat_template:
                q_text = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": "你是一个数学推理验证专家，能够辨别推理步骤中的错误。"}, 
                    {"role": "user", "content": f"{instruction}"}
                ], tokenize=False, add_generation_prompt=True)
            
            # 将文本转换为token ID
            input_ids = self.tokenizer(q_text, return_tensors="pt").input_ids[0]
            
            # 定位推理步骤并提取验证标签
            step_indices, step_verification_labels = locate_reasoning_steps(instruction, self.tokenizer)
            
            # 转换验证标签为张量
            verification_labels = torch.tensor(step_verification_labels, dtype=torch.long) if step_verification_labels else torch.tensor([], dtype=torch.long)
            
            # 创建标签（用于生成损失）
            labels = input_ids.clone()
            
            # 加载并处理图像
            image = Image.open(full_image_path).convert('RGB')
            pixel_values = self.processor(text=None, images=image)['pixel_values'][0]
            
        except Exception as e:
            logger.warning(f"处理第{index}个样本时出错: {e}")
            # 创建默认数据
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values'][0]
            
            # 创建默认输入
            input_ids = self.tokenizer("图像无法加载。", return_tensors="pt").input_ids[0]
            labels = input_ids.clone()
            step_indices = []  # 确保这些键始终存在
            verification_labels = torch.tensor([], dtype=torch.long)
        
        # 返回处理好的样本
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'step_indices': step_indices,  # 确保始终返回这个键
            'verification_labels': verification_labels  # 确保始终返回这个键
        }


# 定义扩展数据收集器，支持验证数据
class PPMDataCollator:
    def __init__(self, tokenizer):
        """
        初始化数据收集器
        参数:
        - tokenizer: 分词器
        """
        self.tokenizer = tokenizer
    
    # 修改数据收集器的__call__方法，添加防御性检查
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将一批样本整合为模型输入格式"""
        # 找出最大序列长度
        max_len = max(len(feature['input_ids']) for feature in features)
        
        input_ids = []
        labels = []
        pixel_values = []
        step_indices = []
        verification_labels = []
        
        # 处理每个样本
        for feature in features:
            # 对input_ids和labels进行填充
            input_ids.append(feature['input_ids'].tolist() + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'].tolist() + [-100] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
            # 安全地获取步骤位置和验证标签
            step_indices.append(feature.get('step_indices', []))
            verification_labels.append(feature.get('verification_labels', torch.tensor([], dtype=torch.long)))
            
        # 返回批处理后的数据
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.stack([x for x in pixel_values]),
            'step_indices': step_indices,
            'verification_labels': verification_labels,
            'attention_mask': (torch.tensor(input_ids) != self.tokenizer.pad_token_id).long()
        }


# 自定义训练器，支持二元验证损失
class PPMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写计算损失方法，支持验证损失
        增加num_items_in_batch参数以匹配Transformers库的最新API
        """
        outputs = model(
            input_ids=inputs['input_ids'],
            labels=inputs['labels'],
            pixel_values=inputs['pixel_values'],
            step_indices=inputs['step_indices'],
            verification_labels=inputs['verification_labels']
        )
        
        # 总损失已在模型内部计算，包含生成损失和验证损失
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# 主函数：模型训练流程
if __name__ == '__main__':
    # 创建模型配置
    config = PPMConfig(
        enable_verification=True,  # 启用验证功能
        freeze_vision_model=True,  # 冻结视觉模型
        freeze_mlp_projections=True,  # 冻结MLP投影层
        freeze_llm_model=False  # 不冻结语言模型
    )
    
    # 加载视觉处理器和文本分词器
    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/multimodal/models/siglip-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/multimodal/models/Qwen2.5-0.5B-Instruct')
    
    # 注册自定义模型配置和模型类
    AutoConfig.register("ppm_model", PPMConfig)
    AutoModelForCausalLM.register(PPMConfig, PPMModel)

    # 注册基础 VLM 模型类型，因为预训练模型是这个类型
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)

    # 从第二阶段SFT模型加载权重
    base_model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/multimodal/train_multimodal_from_scratch/save/geo_sft')

    # 查看PPMModel的__init__方法签名
    # 假设PPMModel继承自VLM，使用与VLM相同的初始化参数
    model = PPMModel(config)

    # 复制base_model的各个组件到新模型
    model.llm_model = base_model.llm_model
    model.vision_model = base_model.vision_model
    model.tokenizer = base_model.tokenizer 
    model.linear1 = base_model.linear1
    model.linear2 = base_model.linear2

    # 添加验证头
    model.verification_head = BinaryPredictionHead(model.llm_model.config.hidden_size)

    # 应用第三阶段配置（冻结视觉和MLP，解冻LLM）
    model.config.enable_verification = True

    # 确保视觉塔和MLP投影层被冻结
    for name, param in model.named_parameters():
        if 'vision_model' in name or 'linear1' in name or 'linear2' in name:
            param.requires_grad = False
        if 'llm_model' in name:
            param.requires_grad = True
        if 'verification_head' in name:
            param.requires_grad = True

    # 将模型移至GPU
    model = model.cuda()

    # 启用梯度检查点
    model.llm_model.gradient_checkpointing_enable()
    
    # 打印模型参数量和可训练参数量
    print(f'模型总参数量: {sum(p.numel() for p in model.parameters())}') 
    print(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}') 
    
    # 设置数据路径
    mathv360k_base_path = '/root/autodl-tmp/multimodal/dataset/MathV360K'
    dualmathv1_path = '/root/autodl-tmp/multimodal/dataset/DualMath-1.1M/train.jsonl'  # 注意扩展名为jsonl
    output_dir = 'save/geo_ppm' 

    # 创建数据集
    full_dataset = ProcessRewardDataset(
        mathv360k_base_path=mathv360k_base_path,
        dualmathv1_path=dualmathv1_path,
        tokenizer=tokenizer,
        processor=processor,
        config=config
    )

    # 从数据集中划分验证集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.05)  # 5%作为验证集
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    
    # 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        # 增加批量大小
        per_device_train_batch_size=4,  # 从2增加到4
        learning_rate=3e-5,
        num_train_epochs=2,
        # 确保save_steps是eval_steps的整数倍
        save_steps=2000,  # 修改为2000，与eval_steps相同
        save_total_limit=2,
        fp16=True,
        # 减少梯度累积步数，加快更新频率
        gradient_accumulation_steps=8,  # 从8减少到4
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        # 增加数据加载线程数
        dataloader_num_workers=8,  # 从4增加到8
        do_eval=True,
        # 使用新参数名，避免警告
        eval_strategy="steps",  # 替换已弃用的evaluation_strategy
        eval_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 添加梯度检查点以节省显存
        gradient_checkpointing=False,
    )
    
    # 创建训练器，包含验证集
    trainer = PPMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PPMDataCollator(tokenizer)
    )
    
    # 开始训练
    trainer.train(resume_from_checkpoint=False)
    
    # 保存模型和训练状态
    trainer.save_model('save/ursa-rm-8b')
    trainer.save_state()