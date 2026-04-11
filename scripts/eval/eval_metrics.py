import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import lpips
import clip
import pandas as pd  

class LPIPS_Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        self.loss_fn_alex.eval()

    #特征提取张量化
    def compute_score(self, source_image, generated_image):
        if isinstance(source_image, str):
            source_image = lpips.load_image(source_image)
            generated_image = lpips.load_image(generated_image)
        img0_tensor = lpips.im2tensor(source_image).to(self.device)
        img1_tensor = lpips.im2tensor(generated_image).to(self.device)
        with torch.no_grad():
            dist = self.loss_fn_alex(img0_tensor, img1_tensor)
        return dist.item()

class CLIP_Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    def compute_image_text_score(self, image_path, target_prompt):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize([target_prompt], truncate=True).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        return similarity

def run_evaluation(model_name, device='cuda'):
    print(f"\n 开始对模型 [{model_name}] 进行打分...\n")

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    config_json_path = os.path.join(project_root, "datasets", "Hybrid-EditBench.json")
    generated_dir = os.path.join(project_root, "datasets", "generated", model_name)

    if not os.path.exists(config_json_path):
        print(" 错误: 找不到数据集配置！")
        return

    lpips_eval = LPIPS_Evaluator(device=device)
    clip_eval = CLIP_Evaluator(device=device)

    with open(config_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    records = []

    for comp, categories in dataset.items():
        print(f"\n 开始评估 [{comp.upper()}] 难度组:")
        for category, items in categories.items():
            if not items: continue 

            #进度条打印
            pbar = tqdm(items, desc=f"  评测任务: {category}")
            for item in pbar:
                image_id = item["id"]
                source_path = os.path.join(project_root, item["image_path"])
                target_prompt = item["target_prompt"] 
                generated_path = os.path.join(generated_dir, comp, category, f"{image_id}.jpg")

            
                if not os.path.exists(generated_path):
                    pbar.write(f"   警告: 找不到生成图片 {generated_path}，已跳过。")
                    continue
                    
                #  计算两项核心指标
                score_lpips = lpips_eval.compute_score(source_path, generated_path)
                score_clip = clip_eval.compute_image_text_score(generated_path, target_prompt)

                records.append({
                    "Method": model_name.upper(),
                    "Level": comp.upper(),
                    "Task": category,             
                    "ImageID": image_id,
                    "LPIPS (↓)": score_lpips,
                    "CLIP (↑)": score_clip
                })

    if not records:
        print(f" 错误: 没有找到任何 [{model_name}] 生成的图片")
        return

    df = pd.DataFrame(records)

    print("\n" + "="*70)
    print(f" [{model_name.upper()}] 最终评测报告:")
    print("="*70)

    # 宏观层级对比 
    print("\n[一，宏观图像复杂度表现]")
    macro_summary = df.groupby("Level")[["LPIPS (↓)", "CLIP (↑)"]].mean().round(4)
    # 添加计数列，方便核对样本数是否正确
    macro_summary['Count'] = df.groupby("Level").size()
    print(macro_summary.to_markdown())

    # 微观任务对比 
    print("\n[二，微观编辑任务表现]")
    micro_summary = df.groupby(["Level", "Task"])[["LPIPS (↓)", "CLIP (↑)"]].mean().round(4)
    micro_summary['Count'] = df.groupby(["Level", "Task"]).size()
    print(micro_summary.to_markdown())
    
    print("\n" + "="*70)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model in ['sd', 'ip2p', 'p2pz']:
        run_evaluation(model_name=model, device=device)
        print("\n\n") # 留白换行