import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from accelerate import Accelerator
import wandb

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/raid/data/models/Qwen3-8B")
    trust_remote_code: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default="/raid/data/datasets/AgentTraj-L/mix_data_4tasks.json")
    max_seq_length: int = field(default=32768)

@dataclass
class TrainingArgs(TrainingArguments):
    output_dir: str = field(default="./output")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    ddp_timeout: int = field(default=3600)
    fsdp: str = field(default="full_shard auto_wrap")
    fsdp_transformer_layer_cls_to_wrap: str = field(default="Qwen2DecoderLayer")
    report_to: str = field(default="wandb")

def load_dataset(data_path: str):
    """加载JSON格式的数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果数据是列表格式，直接使用；如果是字典格式，提取对应字段
    if isinstance(data, list):
        dataset = Dataset.from_list(data)
    else:
        dataset = Dataset.from_dict(data)
    
    return dataset

def preprocess_function(examples, tokenizer, max_seq_length):
    """预处理函数：将对话转换为模型输入格式"""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(len(examples['conversations'])):
        conversation = examples['conversations'][i]
        
        # 构建对话文本
        messages = []
        for turn in conversation:
            role = turn.get('from', turn.get('role', 'user'))
            content = turn.get('value', turn.get('content', ''))
            
            if role in ['human', 'user']:
                messages.append({"role": "user", "content": content})
            elif role in ['gpt', 'assistant']:
                messages.append({"role": "assistant", "content": content})
        
        # 使用tokenizer的chat_template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 创建labels（通常只对assistant的回复计算loss）
        labels = input_ids.copy()
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    
    return model_inputs

def main():
    # 初始化参数
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArgs()
    
    # 初始化wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "default-project"),  # 使用环境变量，如果不存在则使用默认值
        name=os.getenv("WANDB_NAME", "default-name"),  # 使用环境变量，如果不存在则使用默认值
        config={
            "model": model_args.model_name_or_path,
            "max_seq_length": data_args.max_seq_length,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        }
    )
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        use_fast=True,
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 如果支持flash attention
    )
    
    # 启用gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # 加载数据集
    print("Loading dataset...")
    train_dataset = load_dataset(data_args.data_path)
    
    # 预处理数据集
    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(
            examples, 
            tokenizer, 
            data_args.max_seq_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # 初始化Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    print("Saving final model...")
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "final_model"))
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    main()