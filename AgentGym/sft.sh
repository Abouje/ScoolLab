#!/bin/bash

# --- 环境变量设置 ---
export WANDB_API_KEY="7e628906e0a39f2ac07de2fe7ce63662ae19b96c"
export CUDA_VISIBLE_DEVICES="0,1"
timestamp=$(date -d "+8 hours" +%m%d%H%M)
exp_name="sft_${timestamp}"
export WANDB_PROJECT="agentenv"
export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false" # 设为 true 会上传模型到 wandb，非常慢，建议 false

# 设置PyTorch相关环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Accelerate配置文件路径
ACCELERATE_CONFIG="/home/liang/.cache/huggingface/accelerate/default_config.yaml"

# 创建输出目录
OUTPUT_DIR="/raid/data/zyj/models/qwen8b_${exp_name}"
mkdir -p $OUTPUT_DIR

# 打印配置信息
echo "================================"
echo "FSDP SFT Training Configuration"
echo "================================"
echo "Model: /raid/data/models/Qwen3-8B"
echo "Dataset: /raid/data/datasets/AgentTraj-L/mix_data_4tasks.json"
echo "Output: $OUTPUT_DIR"
echo "Max Seq Length: 32768"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "================================"

# 使用accelerate启动训练
accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    /raid/data/zyj/AgentGym/sft.py \
    --model_name_or_path /raid/data/models/Qwen3-8B \
    --data_path /raid/data/datasets/AgentTraj-L/mix_data_4tasks.json \
    --max_seq_length 32768 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 \
    --gradient_checkpointing \
    --ddp_timeout 3600 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "Qwen2DecoderLayer" \
    --report_to wandb \
    --trust_remote_code 

echo "Training finished! Output saved to: $OUTPUT_DIR"