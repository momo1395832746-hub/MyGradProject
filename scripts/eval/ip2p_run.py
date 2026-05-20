import os
import sys
import json
import argparse
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
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n 开始编辑图像 (使用测试集: {os.path.basename(dataset_path)})\n")
    
    for comp, categories in dataset.items():
        print(f"\n======== 当前评测维度: [{comp.upper()}] ========")
        for category, items in categories.items():#为了适应更新得项目结构多一层嵌套循环
            
            # 自动构建带复杂度的多级输出目录
            cat_out_dir = os.path.join(output_dir, comp, category)
            os.makedirs(cat_out_dir, exist_ok=True)
            
            for item in tqdm(items, desc=f"  正在生成: {category}"):
                image_id = item["id"]
                source_path = os.path.join(project_root, item["image_path"])
                out_path = os.path.join(cat_out_dir, f"{image_id}.jpg")

                # 【新增断点续传机制】：如果图片已存在，直接跳过，节省大量时间！
                if os.path.exists(out_path):
                    continue
                
                if not os.path.exists(source_path):
                    tqdm.write(f"   找不到原图 {source_path}，跳过！")
                    continue
                
                init_image = Image.open(source_path).convert("RGB")
                generated_image = editor.edit_image(
                    image=init_image, 
                    prompt=item["instruction"],
                    num_inference_steps=20 
                )
                generated_image.save(out_path)
            
    print("\n InstructPix2Pix 所有任务生成完毕！")
    print(f" 请前往 {output_dir} 查看结果。")

if __name__ == "__main__":
    # 增加命令行参数解析
    parser = argparse.ArgumentParser(description="批量运行图像编辑")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 JSON 文件的绝对或相对路径")
    args = parser.parse_args()
    
    run_batch_evaluation(args.dataset)