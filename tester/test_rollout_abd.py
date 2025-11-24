# /raid/data/zyj/tester/test_rollout_abd.py

import os
import sys
import asyncio
import pickle
from transformers import AutoTokenizer
import json

# --- 1. 设置环境路径和全局变量 ---
sys.path.append('/raid/data/ly/tester/tester')
import aitester as af
from aitester.envs.abd import ABD

# 你的模型和数据路径
BASE_MODEL_PATH = "/raid/data/models/affine-gpt-oss-20b/"
# ABD环境通常使用本地数据集，这里设置默认路径
DATASET_PATH = "/raid/data/ly/tester/tester/data/rl-python.jsonl"

# --- 2. 复制你的核心函数 (prompt_fn, reward_fn) ---

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# 初始化ABD环境
abd_evaluator = ABD()

# ABD环境的系统提示
ABD_SYSTEM_PROMPT = """
You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.

Your task is to analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.

Format the input data like this:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

Requirements for the input data within the tags:
1. Each line of input should be on a separate line
2. Use the exact format the program expects  
3. Provide the raw input values that should be fed to stdin
4. Do not include any prefixes or extra formatting within the INPUT tags
"""

def make_prompt_fn(item: dict) -> str:
    # ABD环境的prompt需要包含program和expected_output
    program = item.get("program", "")
    expected_output = item.get("expected_output", "")
    
    # 构建prompt
    prompt_content = f"""
Program:
```python
{program}
```

Expected Output:
```
{expected_output}
```

Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.
"""
    
    messages = [
        {"role": "system", "content": f"{ABD_SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{prompt_content}"}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def abd_reward_fn(answer_text: str, item: dict) -> float:
    # 创建response对象
    response_obj = af.Response(
        response=answer_text,
        latency_seconds=0.0,  # 在奖励函数中，我们不关心延迟
        attempts=1,           # 假设是第一次尝试
        model="affine-gpt-oss-20b", # 模型名
        error=None,           # 假设没有网络错误
        success=True          # 假设 API 调用是成功的（因为我们已经拿到了 answer_text）
    )
    
    # 创建challenge对象
    challenge_obj = af.Challenge(
        env=abd_evaluator,  # 使用ABD环境
        prompt=item.get("prompt", ""), 
        extra={
            "program": item.get("program", ""),
            "expected_output": item.get("expected_output", "")
        }
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # 评估模型响应
    evaluation_result = loop.run_until_complete(
        abd_evaluator.evaluate(challenge_obj, response_obj)
    )
    
    # 检查是否有错误
    error_type = evaluation_result.extra.get("error")
    if error_type:
        if "No input found" in error_type: 
            return -1.0
        else: 
            return -0.5
    
    # 返回评估分数（1.0表示完全正确，0.0表示错误）
    return evaluation_result.score

# --- 3. 编写独立的 Rollout 测试主函数 ---

async def test_rollout_process():
    print("--- Starting Standalone Rollout Test for ABD Environment ---")

    # 1. 加载一小批数据用于测试
    print(f"Loading challenges from dataset...")
    
    # ABD环境通常使用LocalDataset，我们创建一些测试用例
    test_items = []
    
    # 加载本地JSONL文件或创建示例
    try:
        if os.path.exists(DATASET_PATH):
            import json
            with open(DATASET_PATH, 'r') as f:
                lines = f.readlines()[:5]  # 只读取前5个样本
                for line in lines:
                    item = json.loads(line)
                    test_items.append({
                        "program": item.get("program", ""),
                        "expected_output": item.get("output", ""),
                        "prompt": f"Program: {item.get('program', '')}\nExpected Output: {item.get('output', '')}"
                    })
        else:
            # 如果文件不存在，创建示例测试用例
            print(f"Warning: Dataset file {DATASET_PATH} not found, creating example test cases...")
            
            # 简单的求和程序示例
            sum_program = """
# Read two numbers from input and print their sum
num1 = int(input())
num2 = int(input())
print(num1 + num2)
"""
            test_items.append({
                "program": sum_program,
                "expected_output": "15",
                "prompt": f"Program: {sum_program}\nExpected Output: 15"
            })
            
            # 简单的循环程序示例
            loop_program = """
# Read a number and print numbers from 1 to that number
try:
    n = int(input())
    for i in range(1, n+1):
        print(i)
except:
    print("Invalid input")
"""
            test_items.append({
                "program": loop_program,
                "expected_output": "1\n2\n3",
                "prompt": f"Program: {loop_program}\nExpected Output: 1\n2\n3"
            })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if not test_items:
        print("No test items available. Exiting.")
        return
    
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
        max_tokens=2000  # ABD环境可能需要较少的token
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
            reward = abd_reward_fn(answer_text, item)
            
            print(f"\n----- Rollout #{j+1} -----")
            print(f"  Reward: {reward:.2f}")
            
            # 打印完整的、格式化的模型响应
            print("  Full Model Response:")
            print("  " + "-"*30)
            # 使用 textwrap 缩进，让输出更整齐
            import textwrap
            formatted_response = textwrap.indent(answer_text, prefix="  | ")
            print(formatted_response)
            print("  " + "-"*30)
            
            all_rollout_data.append({
                "challenge_index": i,
                "rollout_index": j,
                "reward": reward,
                "answer": answer_text
            })

    print("\n--- Standalone Rollout Test Finished ---")
    
    # 把结果保存到 json 文件中以便详细分析
    output_file = "rollout_abd_test_results.json"
    with open(output_file, "w") as f:
        json.dump(all_rollout_data, f, indent=2)
    print(f"Results saved to {output_file}")

# --- 6. 运行测试 ---
if __name__ == "__main__":
    # 因为 vLLM 和 abd_evaluator 都用到了 asyncio，所以我们用 async def main
    asyncio.run(test_rollout_process())