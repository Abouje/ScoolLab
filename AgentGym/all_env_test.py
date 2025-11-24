import json
from pathlib import Path
from tqdm import tqdm

from agentenv.controller import APIAgent, Evaluator
from agentenv.envs import (
    WebshopTask,
    AlfWorldTask,
    BabyAITask,
    SciworldTask,
    TextCraftTask,
    ABDTask,
    DEDTask,
    SATTask,
)

from datetime import datetime, timedelta
import os
import traceback

merchine_time = datetime.now()
user_time = merchine_time + timedelta(hours=8)
timestamp = user_time.strftime("%m%d_%H%M")

output_dir = "/raid/data/zyj/AgentGym/eval_results"
os.makedirs(output_dir, exist_ok=True)

# MODEL_PATH = "/raid/data/models/openai/gpt-oss-20b"
# MODEL_PATH = "/raid/data/models/Qwen3-8B"
MODEL_PATH = "/raid/data/zyj/models/behavioral_clone_11210053_fsdp/train_epoch_4"

# 仅对gpt-oss系列模型有效
REASONING_EFFORT = "high" # "low", "medium", "high"

# 仅对Qwen3系列模型有效
QWEN3_TOP_K = 20

Temperature_dict = {
    "/raid/data/models/openai/gpt-oss-20b": 1.0,
    "/raid/data/models/Qwen3-8B": 0.6,
    MODEL_PATH: 0.6,
}

Top_p_dict = {
    "/raid/data/models/openai/gpt-oss-20b": 1.0,
    "/raid/data/models/Qwen3-8B": 0.95,
    MODEL_PATH: 0.95,
}

Context_window_dict = {
    "/raid/data/models/openai/gpt-oss-20b": 131072,
    "/raid/data/models/Qwen3-8B": 131072,
    MODEL_PATH: 32768,
}

agent_kwargs = {
    "api_key": "",
    "base_url": "http://localhost:8081/v1",
    "model": MODEL_PATH,
    "tot_tokens": Context_window_dict[MODEL_PATH],  # 模型总上下文长度
    "temperature": Temperature_dict[MODEL_PATH],
    "top_p": Top_p_dict[MODEL_PATH],
}
if REASONING_EFFORT and "gpt-oss" in MODEL_PATH:
    agent_kwargs["reasoning_effort"] = REASONING_EFFORT
if "qwen3" in MODEL_PATH.lower() or "train_result" in MODEL_PATH.lower():
    agent_kwargs["extra_body"] = {"top_k": QWEN3_TOP_K}

evaluator = Evaluator(
    APIAgent(**agent_kwargs),
    [
        # WebshopTask(
        #     client_args={
        #         "env_server_base": "http://127.0.0.1:36001",
        #         "data_len": 200,
        #         "timeout": 300,
        #     },
        #     n_clients=4,
        # ),
        # AlfWorldTask(
        #     client_args={
        #         "env_server_base": "http://127.0.0.1:36002",
        #         "data_len": 200,
        #         "timeout": 300,
        #     },
        #     n_clients=4,
        # ),
        # BabyAITask(
        #     client_args={
        #         "env_server_base": "http://127.0.0.1:36003",
        #         "data_len": 90,
        #         "timeout": 300,
        #     },
        #     n_clients=4,
        # ),
        # SciworldTask(
        #     client_args={
        #         "env_server_base": "http://127.0.0.1:36004",
        #         "data_len": 200,
        #         "timeout": 300,
        #     },
        #     n_clients=4,
        # ),
        # TextCraftTask(
        #     client_args={
        #         "env_server_base": "http://127.0.0.1:36005",
        #         "data_len": 100,
        #         "timeout": 300,
        #     },
        #     n_clients=4,
        # ),
        # ABDTask(
        #     client_args={
        #         "env_server_base": "http://127.0.0.1:36006",
        #         "data_len": 50,
        #         "timeout": 300,
        #     },
        #     n_clients=4,
        # ),
        DEDTask(
            client_args={
                "env_server_base": "http://127.0.0.1:36007",
                "data_len": 193,
                "timeout": 900,
            },
            n_clients=4,
        ),
        SATTask(
            client_args={
                "env_server_base": "http://127.0.0.1:36008",
                "data_len": 200,
                "timeout": 300,
            },
            n_clients=4,
        ),
    ],
)


def _load_indices(json_path: str) -> list[int]:
    """读取 AgentEval 测试集 JSON，提取 item_id 尾部数字。"""
    data = json.loads(Path(json_path).read_text())
    return [int(item["item_id"].rsplit("_", 1)[-1]) for item in data]

# 评测索引直接读取 AgentEval 固定关卡（其余任务沿用原逻辑）
DATA_ROOT = Path("/raid/data/datasets/AgentEval")
TASK_TO_FILE = {
    "WebShop": DATA_ROOT / "webshop_test.json",
    "AlfWorld": DATA_ROOT / "alfworld_test.json",
    "BabyAI": DATA_ROOT / "babyai_test.json",
    "SciWorld": DATA_ROOT / "sciworld_test.json",
    "TextCraft": DATA_ROOT / "textcraft_test.json",
}

idxs_list = []
for task in evaluator.tasks:
    task_name = getattr(task, "env_name", task.__class__.__name__)
    if task_name in TASK_TO_FILE:
        idxs_list.append(_load_indices(TASK_TO_FILE[task_name]))
    elif task_name in {"ABD", "DED", "SAT"}:
        idxs_list.append(list(range(task.clients[0].data_len)))
    else:
        idxs_list.append([])

# ========== 逐任务评测并即时打印 ==========
all_scores = []
all_exps = []

for task, task_idxs in zip(evaluator.tasks, idxs_list):
    task_name = getattr(task, "env_name", task.__class__.__name__)
    print(f"\n\n==== RUN TASK: {task_name} ====\n", flush=True)

    # 本地模型推理时可传 generation_config；这里沿用全局 max_rounds
    task_exps = []
    total = len(task_idxs)
    batch_size = len(getattr(task, "clients", [])) or 1
    # 使用tqdm显示进度条
    for start in tqdm(range(0, total, batch_size), desc=f"[{task_name}] progress", unit="batch"):
        batch = task_idxs[start : start + batch_size]
        task_exps.extend(
            task.generate_experience(
                agent=evaluator.agent,
                idxs=batch,
                # generation_config=GenerationConfig(...),
                max_rounds=10,
            )
        )
    all_exps.extend(task_exps)
    try:
        task_exps_dicts = []
        for exp in task_exps:
            task_exps_dicts.append({
                'conversation': exp.conversation,
                'reward': exp.reward,
                'error': exp.error
            })
        output_file = os.path.join(output_dir, f"{task_name}_{timestamp}.json")
        with open(output_file, "w") as f:
            json.dump(task_exps_dicts, f, ensure_ascii=False, indent=4)
    except Exception as e:
        error_msg = f"保存任务 {task_name} 结果失败: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())

    # 任务内指标
    scores = [e.reward for e in task_exps]
    n = len(scores)
    all_scores.extend(scores)
    avg_score = sum(scores) / n if n else float("nan")
    success_cnt = sum(1 for score in scores if score == 1 or score == 100)
    success = success_cnt / n if n else float("nan")

    # 打印该任务最后 3 条对话（便于快速抽查）
    # print("\n==== EXPERIENCES (last 3) ====\n")
    # for idx, exp in enumerate(task_exps[-3:]):
    #     print(f"\n\n---- {task_name} EXP {idx} ----\n")
    #     for message in exp.conversation:
    #         role = message.get("role", message.get("from", ""))
    #         content = message.get("content", message.get("value", ""))
    #         if role in ("assistant", "gpt"):
    #             if message.get("reasoning_content") is not None:
    #                 print(f"\n### Reasoning\n{message['reasoning_content']}")
    #             print(f"\n### Agent\n{content}")
    #         else:
    #             print(f"\n### Env\n{content}")

    # 该任务评估结果（即时打印）
    print("\n==== EVALUATION (per task) ====\n")
    print(f"Task: {task_name}")
    print(f"Score: {avg_score:.4f}")
    print(f"Success: {success:.4f} ({success_cnt}/{n})", flush=True)

# ========== 全局聚合 ==========
if all_scores:
    global_score = sum(all_scores) / len(all_scores)
    global_success = sum(1 for score in all_scores if score == 1 or score == 100) / len(all_scores)
else:
    global_score = float("nan")
    global_success = float("nan")


print("\n\n==== OVERALL EVALUATION ====\n")
print(f"Score: {global_score}")
print(f"Success: {global_success}")
