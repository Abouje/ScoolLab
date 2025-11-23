#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES='6,7'

# vllm serve /raid/data/models/openai/gpt-oss-20b \
#   --tensor-parallel-size 2 \
#   --gpu-memory-utilization 0.9 \
#   --async-scheduling \
#   --max-model-len 131072 \
#   --port 8008 \
#   # --tool-call-parser openai \
#   # --enable-auto-tool-choice

# vllm serve /raid/data/models/Qwen3-8B \
#   --tensor-parallel-size 2 \
#   --gpu-memory-utilization 0.9 \
#   --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
#   --max-model-len 131072 \
#   --reasoning-parser qwen3 \
#   --port 8008 \

vllm serve /raid/data/models/Qwen3-8B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --port 8088 \
