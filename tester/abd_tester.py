#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import time, random, asyncio, traceback, contextlib, os
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import aitester as af
from aitester.chutes import get_chute
import json
from datetime import datetime

# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #

@af.cli.command("abd_tester")
def runner():
    async def _run():
        # 创建结果保存目录
        result_dir = "/raid/data/zyj/tester/data"
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成结果文件名，包含时间戳
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        result_file = os.path.join(result_dir, f"abd_test_results_{timestamp}.json")
        
        # 存储所有测试结果
        all_results = {
            "timestamp": timestamp,
            "test_environment": "abd",
            "test_parameters": {},
            "challenges": []
        }
        
        try:
            chute = await get_chute("5e0f5565-8c8c-59a5-bcc9-8f419a4b3c3c")
            print("Chute configuration loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to get chute configuration: {e}")
            chute = None
        
        # --- 1. 定义评测参数 ---
        NUM_UNIQUE_CHALLENGES = 200  # 测试的题目数量
        SAMPLES_PER_CHALLENGE = 1  # 每道题的尝试次数 (pass@k)
        TIMEOUT_PER_RUN = 600  # 每次运行的超时时间（秒）
        
        # 保存测试参数
        all_results["test_parameters"] = {
            "num_unique_challenges": NUM_UNIQUE_CHALLENGES,
            "samples_per_challenge": SAMPLES_PER_CHALLENGE,
            "timeout_per_run": TIMEOUT_PER_RUN,
        }
        
        # --- 2. 存储总体结果 ---
        total_challenges_passed = 0
        start_time = time.time()
        
        # 只测试abd环境
        env_name = "ABD"
        print(f"\nStarting tests for environment: {env_name}")
        
        try:
            # 尝试获取ABD环境类
            env_class = af.ENVS.get(env_name)
            if not env_class:
                raise ValueError(f"Environment '{env_name}' not found in aitester.ENVS")
            
            env_instance = env_class()
            
            # --- 4. 外层循环：运行 NUM_UNIQUE_CHALLENGES 轮 ---
            for challenge_idx in range(1, NUM_UNIQUE_CHALLENGES + 1):
                challenge_start_time = time.time()
                challenge_results = {
                    "challenge_index": challenge_idx,
                    "prompt": "",
                    "attempts": [],
                    "passed": False,
                    "execution_time": 0
                }
                
                print(f"\n{'='*25} STARTING CHALLENGE {challenge_idx}/{NUM_UNIQUE_CHALLENGES} FOR {env_name} {'='*25}")
                
                try:
                    # --- 5. 在每轮开始时，生成一个全新的 challenge ---
                    print("Generating a new unique challenge...")
                    challenge = await env_instance.generate()
                    
                    # 保存prompt
                    challenge_results["prompt"] = challenge.prompt
                    
                    # 打印这道题的 Prompt
                    print("\n" + "-"*20 + f" CHALLENGE #{challenge_idx} PROMPT " + "-"*20)
                    print(challenge.prompt)
                    print("-"*(43 + len(str(challenge_idx))) + "\n")
                    
                    executor = af.Executor(
                        # 使用与原代码相同的模型配置
                        model="/raid/data/models/affine-gpt-oss-20b-bf16/",
                        revision="local-vllm",
                        block=0,
                        chute=chute,
                        slug=chute['slug'] if chute else None
                    )
                    
                    # --- 6. 内层循环：对这个 challenge 进行多次采样 ---
                    print(f"Running {SAMPLES_PER_CHALLENGE} samples for this challenge (pass@{SAMPLES_PER_CHALLENGE} test)...")
                    
                    tasks = [
                        af.run(challenge, executor, timeout=TIMEOUT_PER_RUN) 
                        for _ in range(SAMPLES_PER_CHALLENGE)
                    ]
                    
                    results_list = await asyncio.gather(*tasks)
                    run_results = [item for sublist in results_list for item in sublist]
                    
                    # --- 7. 统计和分析本轮的结果 ---
                    successful_attempts_this_round = 0
                    
                    for i, result in enumerate(run_results):
                        attempt_result = {
                            "attempt_index": i + 1,
                            "response": {
                                "success": result.response.success,
                                "content": result.response.response if result.response.success else None,
                                "error": result.response.error if not result.response.success else None
                            },
                            "evaluation": {
                                "score": result.evaluation.score,
                                "extra": result.evaluation.extra
                            },
                            "passed": False
                        }
                        
                        print(f"----- Attempt {i+1}/{SAMPLES_PER_CHALLENGE} -----")
                        
                        # 打印模型响应
                        print("  [MODEL RESPONSE]:")
                        if result.response.success:
                            import textwrap
                            wrapped_response = textwrap.fill(result.response.response, width=120, initial_indent='    ', subsequent_indent='    ')
                            print(wrapped_response)
                        else:
                            print(f"    ERROR: API call failed. Details: {result.response.error}")
                        
                        # 打印评估结果
                        score = result.evaluation.score
                        print(f"  [EVALUATION]:")
                        print(f"    Score = {score:.4f}")
                        if result.evaluation.extra:
                            print(f"    Details: {json.dumps(result.evaluation.extra, indent=2, ensure_ascii=False)}")
                        
                        if score == 1.0:
                            successful_attempts_this_round += 1
                            attempt_result["passed"] = True
                            print("  [STATUS]: SUCCESS")
                        else:
                            print("  [STATUS]: FAILED")
                        print("-" * (21 + len(str(i+1)) + len(str(SAMPLES_PER_CHALLENGE))))
                        
                        # 保存尝试结果
                        challenge_results["attempts"].append(attempt_result)
                    
                    # --- 8. 总结本轮的 pass@k 结果 ---
                    print(f"\n----- Summary for Challenge #{challenge_idx} -----")
                    if successful_attempts_this_round > 0:
                        print(f"PASS: The model found a solution in {successful_attempts_this_round} out of {SAMPLES_PER_CHALLENGE} attempts.")
                        total_challenges_passed += 1
                        challenge_results["passed"] = True
                    else:
                        print(f"FAIL: The model failed to find a solution in all {SAMPLES_PER_CHALLENGE} attempts.")
                    
                    # 计算执行时间
                    challenge_results["execution_time"] = time.time() - challenge_start_time
                    print(f"Challenge execution time: {challenge_results['execution_time']:.2f} seconds")
                    print(f"{'='*25} END OF CHALLENGE {challenge_idx}/{NUM_UNIQUE_CHALLENGES} FOR {env_name} {'='*25}\n")
                    
                    # 保存挑战结果
                    all_results["challenges"].append(challenge_results)
                    
                    # 定期保存结果
                    if challenge_idx % 10 == 0:
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(all_results, f, indent=2, ensure_ascii=False)
                        print(f"Results saved to {result_file} (partial save after {challenge_idx} challenges)")
                    
                except Exception as e:
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    challenge_results["error"] = error_info
                    print(f"An error occurred during challenge #{challenge_idx} for {env_name}:")
                    traceback.print_exc()
                    all_results["challenges"].append(challenge_results)
            
        except Exception as e:
            print(f"An error occurred when setting up environment {env_name}:")
            traceback.print_exc()
            all_results["error"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
        
        # 计算总体统计信息
        total_execution_time = time.time() - start_time
        all_results["summary"] = {
            "total_challenges": NUM_UNIQUE_CHALLENGES,
            "passed_challenges": total_challenges_passed,
            "pass_rate": total_challenges_passed / NUM_UNIQUE_CHALLENGES if NUM_UNIQUE_CHALLENGES > 0 else 0,
            "total_execution_time": total_execution_time,
            "avg_challenge_time": total_execution_time / NUM_UNIQUE_CHALLENGES if NUM_UNIQUE_CHALLENGES > 0 else 0
        }
        
        # --- 9. 所有轮次结束后的最终总结 ---
        print("\n" + "#"*25 + " FINAL REPORT " + "#"*25)
        print(f"Tested environment: {env_name}")
        print(f"Total unique challenges: {NUM_UNIQUE_CHALLENGES}")
        print(f"Samples per challenge (pass@k): {SAMPLES_PER_CHALLENGE}")
        print("-" * 64)
        print(f"Overall Pass Rate (pass@{SAMPLES_PER_CHALLENGE}): {total_challenges_passed} / {NUM_UNIQUE_CHALLENGES} = {(total_challenges_passed / NUM_UNIQUE_CHALLENGES):.2%}")
        print(f"Total execution time: {total_execution_time:.2f} seconds")
        print(f"Average challenge time: {total_execution_time / NUM_UNIQUE_CHALLENGES:.2f} seconds")
        print(f"Results saved to: {result_file}")
        print("#" * 64)
        
        # 保存最终结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    async def main():
        # 延长 watchdog 超时时间以支持长时间运行的测试
        await asyncio.gather(_run(), af.watchdog(timeout=3600))  # 1小时超时

    asyncio.run(main())

if __name__ == "__main__":
    runner()