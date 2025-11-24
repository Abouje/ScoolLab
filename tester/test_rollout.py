# /data2/ly/hvm_finetuning/test_rollout.py

import os
import sys
import asyncio
import pickle
from transformers import AutoTokenizer
import json

# --- 1. 设置环境路径和全局变量 ---
sys.path.append('/raid/data/ly/tester/tester')
import aitester as af
from aitester.envs.hvm import HVM

# 你的模型和数据路径
#BASE_MODEL_PATH = "/data2/ly/models/affine-gpt-oss-20b"
#BASE_MODEL_PATH = "/data2/ly/models/affine-gpt-oss-20b-clean/"
#BASE_MODEL_PATH ='/data2/Qwen/DeepSeek-R1-Distill-Qwen-7B/'
BASE_MODEL_PATH = "/raid/data/models/affine-gpt-oss-20b/"
DATASET_PATH = "/raid/data/ly/data/hvm/samling_06/data/hvm_challenges_intermediate.pkl"

# --- 2. 复制你的核心函数 (prompt_fn, reward_fn) ---
#    (我们直接从 run_rl_hvm.py 中复制过来)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
hvm_evaluator = HVM()

FINAL_FORMAT_INSTRUCTION = """
You are an expert at solving Hole-filled Virtual Machine (HVM) puzzles.
Your task is to analyze the provided HVM program and test cases to determine the correct integer values for the unknown constants (holes).
Think step-by-step to deduce the values, then, provide the final mapping of all holes within a `<HOLES>` block.
Example:
<HOLES>
?a=...
?b=...
?c=...
...
</HOLES>
"""

def make_prompt_fn(item: dict) -> str:
    problem_description = item["prompt"]
    messages = [
        {"role": "system", "content": f"{FINAL_FORMAT_INSTRUCTION}"},
        {"role": "user", "content": f"{problem_description}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def hvm_reward_fn(answer_text: str, item: dict) -> float:
    response_obj = af.Response(
        response=answer_text,
        latency_seconds=0.0,  # 在奖励函数中，我们不关心延迟
        attempts=1,           # 假设是第一次尝试
        model="affine-gpt-oss-20b", # 模型名
        error=None,           # 假设没有网络错误
        success=True          # 假设 API 调用是成功的（因为我们已经拿到了 answer_text）
    )
    challenge_obj = af.Challenge(
        env=hvm_evaluator,  # <--- 使用 'env' 参数，而不是 'env_name'
        prompt=item["prompt"], 
        extra=item["extra"]
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    evaluation_result = loop.run_until_complete(
        hvm_evaluator.evaluate(challenge_obj, response_obj)
    )
    error_type = evaluation_result.extra.get("error")
    if error_type:
        if "Missing or invalid <HOLES> block" in error_type: return -1.0
        elif "Not all holes provided" in error_type: return -0.5
        else: return -0.2
    return evaluation_result.score if evaluation_result.score == 1.0 else 0.0

# --- 3. 编写独立的 Rollout 测试主函数 ---

async def test_rollout_process():
    print("--- Starting Standalone Rollout Test ---")

    # 1. 加载一小批数据用于测试
    print(f"Loading challenges from {DATASET_PATH}...")
    with open(DATASET_PATH, "rb") as f:
        all_challenges = pickle.load(f)
    
    # 我们只取前2个问题进行测试，每个问题生成4个答案
    test_items = all_challenges[3:10]
    ROLLOUT_NUM = 1
    
    print(f"Will test with {len(test_items)} challenges, generating {ROLLOUT_NUM} rollouts each.")

    # 2. 初始化 vLLM
    print(f"\nInitializing vLLM with model: {BASE_MODEL_PATH}...")
    try:
        from vllm import LLM, SamplingParams
        # 在 A100 上，我们可以直接加载，无需担心显存
        llm = LLM(model=BASE_MODEL_PATH, trust_remote_code=True) 
        print("vLLM Engine initialized successfully!")
    except Exception as e:
        print("\nFATAL: Failed to initialize vLLM Engine.")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return # 如果 vLLM 初始化失败，直接退出

    # 3. 准备 prompts 和采样参数
    prompts = [make_prompt_fn(item) for item in test_items]
    sampling_params = SamplingParams(
        n=ROLLOUT_NUM,          # 每个 prompt 生成 n 个输出
        temperature=0,
        max_tokens=9000
    )

    # 4. 执行生成 (Rollout)
    print("\nExecuting generation (performing rollouts)...")
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete.")

    # 5. 处理和评估结果 (模拟奖励计算)
    print("\n--- Processing and Evaluating Rollout Results ---")
    all_rollout_data = []
    
    for i, item in enumerate(test_items):
        print(f"\n{'='*20} Results for Challenge #{i+1} {'='*20}")
        
        request_output = outputs[i]
        
        for j, completion_output in enumerate(request_output.outputs):
            answer_text = completion_output.text
            reward = hvm_reward_fn(answer_text, item)
            
            print(f"\n----- Rollout #{j+1} -----")
            print(f"  Reward: {reward:.2f}")
            
            # --- 核心修改：打印完整的、格式化的模型响应 ---
            print("  Full Model Response:")
            print("  " + "-"*30)
            # 使用 textwrap 缩进，让输出更整齐
            import textwrap
            formatted_response = textwrap.indent(answer_text, prefix="  | ")
            print(formatted_response)
            print("  " + "-"*30)
            # --- 结束修改 ---
            
            all_rollout_data.append({
                "challenge_index": i,
                "rollout_index": j,
                "reward": reward,
                "answer": answer_text
            })

    print("\n--- Standalone Rollout Test Finished ---")
    
    # 你可以取消下面的注释，把结果保存到 json 文件中以便详细分析
    with open("rollout_test_results.json", "w") as f:
        json.dump(all_rollout_data, f, indent=2)

# --- 6. 运行测试 ---
if __name__ == "__main__":
    # 因为 vLLM 和 hvm_evaluator 都用到了 asyncio，所以我们用 async def main
    asyncio.run(test_rollout_process())