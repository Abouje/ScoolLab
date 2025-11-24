import os
import json

def merge_jsonl_files(target_files, output_file):
    """
    合并目录中的所有jsonl文件到一个输出文件
    
    参数:
    target_files: 包含jsonl文件list
    output_file: 合并后的输出文件路径
    """
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_files = len(target_files)
    print(f"找到 {total_files} 个jsonl文件需要合并")
    
    # 合并文件
    total_lines = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, file_name in enumerate(target_files, 1):
                print(f"处理文件 {i}/{total_files}: {file_name}")
                
                try:
                    with open(file_name, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            # 确保行不为空
                            line = line.strip()
                            if line:
                                # 验证JSON格式
                                try:
                                    json.loads(line)
                                    out_f.write(line + '\n')
                                    total_lines += 1
                                except json.JSONDecodeError as e:
                                    print(f"警告: 文件 {file_name} 中的某行不是有效的JSON: {e}")
                except Exception as e:
                    print(f"错误: 处理文件 {file_name} 时出错: {e}")
        
        print(f"合并完成! 总共处理了 {total_lines} 行数据")
        print(f"合并后的文件保存在: {output_file}")
        return True
    except Exception as e:
        print(f"错误: 写入输出文件时出错: {e}")
        return False

if __name__ == "__main__":
    # 设置源目录和输出文件路径
    target_files = [
        "/raid/data/datasets/AgentTraj-L/qwen3_format/alfworld_general_cleaned.jsonl",
        "/raid/data/datasets/AgentTraj-L/qwen3_format/babyai_general.cleaned.jsonl",
        "/raid/data/datasets/AgentTraj-L/qwen3_format/sciworld_general.jsonl",
        "/raid/data/datasets/AgentTraj-L/qwen3_format/textcraft_general_cleaned.jsonl",
        "/raid/data/datasets/AgentTraj-L/qwen3_format/webshop_general.jsonl",
    ]
    OUTPUT_FILE = '/raid/data/zyj/AgentGym/merged_agent_traj.jsonl'
    
    # 执行合并
    success = merge_jsonl_files(target_files, OUTPUT_FILE)
    
    if not success:
        print("合并过程中出现错误，请检查日志")
    else:
        print("合并操作成功完成!")