#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import time, random, asyncio, traceback, contextlib
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import aitester as af
from aitester.chutes import get_chute
import json


# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #
# ... (import 不变) ...

# aitester/tester.py

# ... (文件顶部的 import 保持不变) ...

@af.cli.command("tester")
def runner():
    async def _run():
        
        chute = await get_chute("5e0f5565-8c8c-59a5-bcc9-8f419a4b3c3c")
        print(json.dumps(chute, indent=4))
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!                           核心修改区域开始                      !!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # --- 1. 定义评测参数 ---
        NUM_UNIQUE_CHALLENGES = 50  # 我们要测试 5 道不同的题
        SAMPLES_PER_CHALLENGE = 1  # 每道题尝试 5 次 (pass@5)

        # --- 2. 存储总体结果 ---
        total_challenges_passed = 0

        for env_class in af.ENVS.values():
            env_name = env_class.__name__
            
            # --- 3. 筛选要进行 pass@k 测试的环境 ---
            if env_name not in ["HVM"]:
                print(f"\nSkipping environment: {env_name}")
                continue

            env_instance = env_class()
            
            # --- 4. 外层循环：运行 NUM_UNIQUE_CHALLENGES 轮 ---
            for challenge_idx in range(1, NUM_UNIQUE_CHALLENGES + 1):
                print(f"\n{'='*25} STARTING CHALLENGE {challenge_idx}/{NUM_UNIQUE_CHALLENGES} FOR {env_name} {'='*25}")
                
                try:
                    # --- 5. 在每轮开始时，生成一个全新的 challenge ---
                    print("Generating a new unique challenge...")
                    challenge = await env_instance.generate()

                    # 打印这道题的 Prompt
                    print("\n" + "-"*20 + f" CHALLENGE #{challenge_idx} PROMPT " + "-"*20)
                    print(challenge.prompt)
                    print("-"*(43 + len(str(challenge_idx))) + "\n")

                    executor = af.Executor(
                        #model="/data2/ly/models/gpt-oss-20b/",
                        model="/raid/data/models/affine-gpt-oss-20b-bf16/",
                        revision="local-vllm",
                        block=0,
                        chute=chute,
                        slug=chute['slug']
                    )

                    # --- 6. 内层循环：对这个 challenge 进行多次采样 ---
                    print(f"Running {SAMPLES_PER_CHALLENGE} samples for this challenge (pass@{SAMPLES_PER_CHALLENGE} test)...")
                    
                    tasks = [
                        af.run(challenge, executor, timeout=600) 
                        for _ in range(SAMPLES_PER_CHALLENGE)
                    ]
                    
                    results_list = await asyncio.gather(*tasks)
                    all_results = [item for sublist in results_list for item in sublist]

                    # --- 7. 统计和分析本轮的结果 ---
                    successful_attempts_this_round = 0
                    
                    for i, result in enumerate(all_results):
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
                            print("  [STATUS]: SUCCESS")
                        else:
                            print("  [STATUS]: FAILED")
                        print("-" * (21 + len(str(i+1)) + len(str(SAMPLES_PER_CHALLENGE))))

                    # --- 8. 总结本轮的 pass@k 结果 ---
                    print(f"\n----- Summary for Challenge #{challenge_idx} -----")
                    if successful_attempts_this_round > 0:
                        print(f"PASS: The model found a solution in {successful_attempts_this_round} out of {SAMPLES_PER_CHALLENGE} attempts.")
                        total_challenges_passed += 1
                    else:
                        print(f"FAIL: The model failed to find a solution in all {SAMPLES_PER_CHALLENGE} attempts.")
                    print(f"{'='*25} END OF CHALLENGE {challenge_idx}/{NUM_UNIQUE_CHALLENGES} FOR {env_name} {'='*25}\n")


                except Exception as e:
                    print(f"An error occurred during challenge #{challenge_idx} for {env_name}:")
                    traceback.print_exc()

        # --- 9. 所有轮次结束后的最终总结 ---
        print("\n" + "#"*25 + " FINAL REPORT " + "#"*25)
        print(f"Tested environments: HVM, ELR")
        print(f"Total unique challenges per environment: {NUM_UNIQUE_CHALLENGES}")
        print(f"Samples per challenge (pass@k): {SAMPLES_PER_CHALLENGE}")
        print("-" * 64)
        print(f"Overall Pass Rate (pass@{SAMPLES_PER_CHALLENGE}): {total_challenges_passed} / {NUM_UNIQUE_CHALLENGES} = {(total_challenges_passed / NUM_UNIQUE_CHALLENGES):.2%}")
        print("#" * 64)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!                           核心修改区域结束                      !!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    async def main():
        await asyncio.gather(_run(), af.watchdog(timeout=2000)) # 延长 watchdog 到 2 小时

    asyncio.run(main())