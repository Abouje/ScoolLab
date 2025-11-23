
export WANDB_API_KEY="7e628906e0a39f2ac07de2fe7ce63662ae19b96c"
export CUDA_VISIBLE_DEVICES="0,1"
timestamp=$(date -d "+8 hours" +%m%d%H%M)
exp_name="behavioral_clone_${timestamp}_fsdp"

n_epochs='4'

# accelerator config
num_processes='2'
main_process_port='8897'
config_file="/home/liang/.cache/huggingface/accelerate/default_config.yaml"

# training arguments
train_file='/raid/data/datasets/AgentTraj-L/mix_data_4tasks.json'
model_train_path="/raid/data/models/Qwen3-8B"
model_save_path="/raid/data/zyj/models/${exp_name}/"

batch_size="8"
eval_batch_size="1"
gradient_accumulation_steps="2"
max_input_length="4096"
num_workers="8"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"

logging_epoch_freq="1"
evaluating_epoch_freq="100"
saving_epoch_freq="1"
logging_step_freq="5"

# wandb config
wandb_log="True"
wandb_project="agentenv"
wandb_run_name="${exp_name}"

# environment parameters
data_len="200"
timeout="2400"

# eval
task_list=("webshop" "alfworld" "babyai" "sciworld" "textcraft")
# eval parameters
temperature="1.0"
max_round_list=("6" "50" "50" "100" "20")
env_server_base_list=("http://127.0.0.1:36001" "http://127.0.0.1:36002" "http://127.0.0.1:36003" "http://127.0.0.1:36004" "http://127.0.0.1:36005")

mkdir -p "${model_save_path}"
# step1: train
accelerate launch \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    /raid/data/zyj/AgentGym/bc.py \
        --train_file "${train_file}" \
        --model_train_path "${model_train_path}" \
        --model_save_path "${model_save_path}" \
        --task_name "${task_list[1]}" \
        --batch_size "${batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --evaluating_epoch_freq "${evaluating_epoch_freq}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_round "${max_round_list[1]}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --env_server_base "${env_server_base_list[1]}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"
