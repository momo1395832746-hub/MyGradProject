import os
import sys
import json
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.editors import SDEditEditor

def run_batch_evaluation():
    print("正在加载 SDEdit...")
    editor = SDEditEditor(device="cuda") 

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    config_path = os.path.join(project_root, "datasets", "Hybrid-EditBench.json")
    output_dir = os.path.join(project_root, "datasets", "generated", "sd")
    
    with open(config_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("\n开始编辑图像\n")
    
    for comp, categories in dataset.items():
        print(f"\n======== 当前评测维度: [{comp.upper()}] ========")
        for category, items in categories.items():
            
            cat_out_dir = os.path.join(output_dir, comp, category)
            os.makedirs(cat_out_dir, exist_ok=True)
            
            for item in tqdm(items, desc=f"  正在生成: {category}"):
                image_id = item["id"]
                source_path = os.path.join(project_root, item["image_path"])
                out_path = os.path.join(cat_out_dir, f"{image_id}.jpg")
                
                if not os.path.exists(source_path):
                    tqdm.write(f"找不到原图 {source_path}，跳过！")
                    continue
                
                init_image = Image.open(source_path).convert("RGB")
                generated_image = editor.edit_image(
                    image=init_image, 
                    prompt=item["instruction"],
                    num_inference_steps=50 
                )
                generated_image.save(out_path)
            
    print("\n InstructPix2Pix 所有任务生成完毕！")
    print(f"请前往 {output_dir} 查看结果。")

if __name__ == "__main__":
    run_batch_evaluation()