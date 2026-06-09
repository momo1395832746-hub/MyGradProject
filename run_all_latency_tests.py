import os
import subprocess
import sys

# 自动获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
# 默认使用完整的基准测试集进行 100 条数据的测速
dataset_path = os.path.join(project_root, "datasets", "Hybrid-EditBench.json")

# 注册需要测速的模型脚本
scripts_to_run = [
    ("InstructPix2Pix", "scripts/eval/ip2p_run.py"),
    ("Pix2Pix-Zero", "scripts/eval/p2pz_run.py"),
    ("SDEdit", "scripts/eval/sd_run.py"),
    ("Qwen-Edit", "scripts/eval/qwen_edit_run.py")
]

print(" 启动全模型计算复杂度 (延迟) 自动化测算流水线...")

for model_name, script_rel_path in scripts_to_run:
    script_full_path = os.path.join(project_root, script_rel_path)
    
    print(f"\n{'='*50}")
    print(f" 正在启动 [{model_name}] 的延迟测试...")
    print(f"{'='*50}")
    
    if not os.path.exists(script_full_path):
        print(f"[-] 找不到脚本 {script_full_path}，已跳过。")
        continue
        
    try:
        # 使用 subprocess 唤醒子进程执行测速
        subprocess.run(
            [sys.executable, script_full_path, "--dataset", dataset_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[x] {model_name} 测速中断: {e}")
        print("请检查报错信息，然后重新运行本脚本。")

print("\n 全线模型延迟测算完毕！")
print(" 数据已安全保存至 datasets/results/latency/ 目录，前端系统现在可以读取了。")