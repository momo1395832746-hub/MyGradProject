import os
import sys
import json
import argparse
import time         # 【新增】引入时间模块
import torch        # 【新增】引入torch用于CUDA同步
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.editors import InstructPix2PixEditor

def run_batch_evaluation(dataset_path):
    print(" 正在加载 InstructPix2Pix ...")
    editor = InstructPix2PixEditor(device="cuda") 

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    #config_path = os.path.join(project_root, "datasets", "Hybrid-EditBench.json")

    output_dir = os.path.join(project_root, "datasets", "generated", "ip2p")

    # 【新增】存放延迟数据的目录
    latency_dir = os.path.join(project_root, "datasets", "results", "latency")
    os.makedirs(latency_dir, exist_ok=True)
    latency_records = {} # 记录每张图的耗时
    processed_count = 0  # 计数器
    max_test_size = 100  # 最大测试数量
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n 开始编辑图像 (使用测试集: {os.path.basename(dataset_path)})\n")
    
    try:
        for comp, categories in dataset.items():
            for category, items in categories.items():
                cat_out_dir = os.path.join(output_dir, comp, category)
                os.makedirs(cat_out_dir, exist_ok=True)
                
                for item in tqdm(items, desc=f"  正在生成: {category}"):
                    if processed_count >= max_test_size:
                        raise StopIteration # 利用异常跳出多重循环
                        
                    image_id = item["id"]
                    source_path = os.path.join(project_root, item["image_path"])
                    out_path = os.path.join(cat_out_dir, f"{image_id}.jpg")

                    # 注意：如果为了测时间，建议强制重新生成，或者确保你要测的图还没生成
                    # 如果只想测还没生成的，保留下面这两行；如果想重新跑100张，请注释掉这两行！
                    #if os.path.exists(out_path):
                         #continue
                    
                    if not os.path.exists(source_path):
                        continue
                    
                    init_image = Image.open(source_path).convert("RGB")
                    
                    # 【核心耗时测量区间 - 开始】
                    torch.cuda.synchronize() # 确保之前的CUDA任务都已完成
                    start_time = time.time()
                    
                    generated_image = editor.edit_image(
                        image=init_image, 
                        prompt=item["instruction"],
                        num_inference_steps=20 
                    )
                    
                    torch.cuda.synchronize() # 确保当前生成任务已完成
                    end_time = time.time()
                    # 【核心耗时测量区间 - 结束】
                    
                    latency = end_time - start_time
                    latency_records[image_id] = latency
                    processed_count += 1
                    
                    generated_image.save(out_path)
                    
    except StopIteration:
        print(f"\n 已达到设定的测试上限 {max_test_size} 张图。")
            
    # 【新增】保存延迟数据
    latency_file = os.path.join(latency_dir, "ip2p_latency.json")
    with open(latency_file, "w", encoding="utf-8") as f:
        json.dump(latency_records, f, indent=4)
        
    print("\n InstructPix2Pix 所有任务生成完毕！")
    print(f" 请前往 {output_dir} 查看结果。")
    print(f" 延迟数据已保存在: {latency_file}")

if __name__ == "__main__":
    # 增加命令行参数解析
    parser = argparse.ArgumentParser(description="批量运行图像编辑")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 JSON 文件的绝对或相对路径")
    args = parser.parse_args()
    
    run_batch_evaluation(args.dataset)