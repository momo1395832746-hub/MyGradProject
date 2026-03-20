import os
import sys
import json
import argparse
import time         # 【新增】时间模块
import torch        # 【新增】PyTorch用于CUDA同步
from PIL import Image
from tqdm import tqdm

# 确保脚本能找到项目根目录下的 editors.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.editors import Pix2PixZeroEditor

def run_batch_evaluation(dataset_path):
    print(" 正在加载 Pix2PixZeroEditor...")

    editor = Pix2PixZeroEditor(device="cuda")

    current_file_path = os.path.abspath(__file__)
    scripts_eval_dir = os.path.dirname(current_file_path)
    scripts_dir = os.path.dirname(scripts_eval_dir)
    project_root = os.path.dirname(scripts_dir)
    
    output_dir = os.path.join(project_root, "datasets", "generated", "p2pz") 
    # 【新增】存放延迟数据的目录
    latency_dir = os.path.join(project_root, "datasets", "results", "latency")
    os.makedirs(latency_dir, exist_ok=True)
    latency_records = {} 
    processed_count = 0  
    max_test_size = 100
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("\n 开始编辑图像\n")
    
    try:
        for comp, categories in dataset.items():
            print(f"\n======== 当前评测维度: [{comp.upper()}] ========")
            
            for category, items in categories.items():
                
                cat_out_dir = os.path.join(output_dir, comp, category)
                os.makedirs(cat_out_dir, exist_ok=True)
                
                for item in tqdm(items, desc=f"  正在生成: {category}"):
                    if processed_count >= max_test_size:
                        raise StopIteration

                    image_id = item["id"]
                    source_path = os.path.join(project_root, item["image_path"])

                    source_prompt = item["source_prompt"]
                    target_prompt = item["target_prompt"]
                    
                    out_path = os.path.join(cat_out_dir, f"{image_id}.jpg")
                    
                    # 注意：如果为了测时间，建议强制重新生成，或者确保你要测的图还没生成
                    #if os.path.exists(out_path):
                        #continue

                    if not os.path.exists(source_path):
                        tqdm.write(f"找不到原图 {source_path}，跳过！")
                        continue
                        
                    init_image = Image.open(source_path).convert("RGB")
                    
                    # 【核心耗时测量区间 - 开始】
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    generated_image = editor.edit_image(
                        image=init_image, 
                        prompt=target_prompt,        
                        source_prompt=source_prompt, 
                        seed=42,                     
                        num_inference_steps=50,      
                        cross_attention_guidance_amount=0.10 # 注意力引导强度
                    )
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    # 【核心耗时测量区间 - 结束】
                    
                    latency_records[image_id] = end_time - start_time
                    processed_count += 1
                    
                    generated_image.save(out_path)
                    
    except StopIteration:
        print(f"\n 已达到设定的测试上限 {max_test_size} 张图。")

    # 【新增】保存延迟数据
    latency_file = os.path.join(latency_dir, "p2pz_latency.json")
    with open(latency_file, "w", encoding="utf-8") as f:
        json.dump(latency_records, f, indent=4)
            
    print("\n P2PZ 所有任务生成完毕！")
    print(f"请前往 {output_dir} 查看结果。")
    print(f" 延迟数据已保存在: {latency_file}")

if __name__ == "__main__":
    # 增加命令行参数解析
    parser = argparse.ArgumentParser(description="批量运行图像编辑")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 JSON 文件的绝对或相对路径")
    args = parser.parse_args()
    
    run_batch_evaluation(args.dataset)