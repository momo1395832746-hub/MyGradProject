import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import lpips
import pandas as pd  
from transformers import CLIPProcessor, CLIPModel # 引入 HF 的 CLIP 库以支持 D-CLIP

class LPIPS_Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        print(f" 初始化 LPIPS 引擎，使用设备: {self.device}")
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        self.loss_fn_alex.eval()

    # 特征提取张量化
    def compute_score(self, source_image, generated_image):
        if isinstance(source_image, str):
            source_image = lpips.load_image(source_image)
            generated_image = lpips.load_image(generated_image)
        img0_tensor = lpips.im2tensor(source_image).to(self.device)
        img1_tensor = lpips.im2tensor(generated_image).to(self.device)

        # 🚀 【学术级严谨对齐】：让 source_tensor 向 generated_tensor 妥协
        if img0_tensor.shape != img1_tensor.shape:
            # 记录日志，方便你后期在论文中统计“API 擅自改尺寸的频率”
            # print(f"尺寸不匹配: 原图 {img0_tensor.shape[2:]} vs 生成图 {img1_tensor.shape[2:]}")
            
            # 使用更平滑的 bicubic 插值，将原图缩放到生成图的尺寸
            img0_tensor = torch.nn.functional.interpolate(
                img0_tensor, 
                size=img1_tensor.shape[2:], # 以生成图尺寸为基准
                mode='bicubic', 
                align_corners=False
            )
            
        with torch.no_grad():
            dist = self.loss_fn_alex(img0_tensor, img1_tensor)
        return dist.item()

class DCLIP_Evaluator:
    def __init__(self, device='cuda', model_id="openai/clip-vit-base-patch32"):
        """
        初始化 D-CLIP 裁判引擎
        """
        self.device = device
        print(f" 初始化 D-CLIP 引擎，使用设备: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        
        # 【装甲修复】：如果 features 不是纯 Tensor，说明被裹了一层外壳，手动剥离
        if not isinstance(features, torch.Tensor):
            if hasattr(features, 'image_embeds'):
                features = features.image_embeds
            elif hasattr(features, 'pooler_output'):
                features = features.pooler_output
            else:
                features = features[0] # 保底方案：取元组的第一个元素
                
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_text_features(self, text: str):
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        
        # 【装甲修复】：剥离文本特征的复杂外壳
        if not isinstance(features, torch.Tensor):
            if hasattr(features, 'text_embeds'):
                features = features.text_embeds
            elif hasattr(features, 'pooler_output'):
                features = features.pooler_output
            else:
                features = features[0]
                
        return features / features.norm(dim=-1, keepdim=True)

    def calculate_dclip(self, src_image_path: str, edit_image_path: str, 
                        src_prompt: str = None, tgt_prompt: str = None, instruction: str = None):
        """
        计算 D-CLIP 分数，内置动态路由机制
        """
        src_image = Image.open(src_image_path).convert("RGB")
        edit_image = Image.open(edit_image_path).convert("RGB")

        # 1. 计算图像方向向量 (Delta I)
        src_img_feat = self.get_image_features(src_image)
        edit_img_feat = self.get_image_features(edit_image)
        
        delta_I = edit_img_feat - src_img_feat
        delta_I = delta_I / delta_I.norm(dim=-1, keepdim=True)

        # 2. 动态路由：计算文本方向向量 (Delta T)
        if src_prompt and tgt_prompt:
            # 路由 A: 双 Prompt 差值模式
            src_txt_feat = self.get_text_features(src_prompt)
            tgt_txt_feat = self.get_text_features(tgt_prompt)
            delta_T = tgt_txt_feat - src_txt_feat
            delta_T = delta_T / delta_T.norm(dim=-1, keepdim=True)
            
        elif instruction:
            # 路由 B: 单 Instruction 直接映射模式
            delta_T = self.get_text_features(instruction)
            
        else:
            # 若 JSON 缺失文本字段，返回 0 以免中断批量跑图
            return 0.0

        # 3. 计算余弦相似度
        dclip_score = F.cosine_similarity(delta_I, delta_T)
        return dclip_score.item()


import os
import json
import pandas as pd
from tqdm import tqdm

def run_evaluation(model_name, dataset_name, dataset_path, device='cuda', model_display_name=None):
    """
    增量修补版 (Delta Update): 自动跳过已计算图像，仅计算遗漏项，并重组全局平均分。
    """
    print(f"\n 开始对模型 [{model_name.upper()}] 进行深度打分...")

    # 1. 路径初始化
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    generated_dir = os.path.join(project_root, "datasets", "generated", model_name)
    
    # 结果保存路径
    save_path = os.path.join(project_root, 'datasets', 'results', f"{model_name}_{dataset_name}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f" 错误: 找不到数据集配置 {dataset_path}")
        return None, None

    # ==========================================
    # 🎯 核心补丁 1/2：读取历史战报，建立免考名单
    # ==========================================
    calculated_records = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                # 必须依赖我们在下方新增的 raw_records 字段
                if "raw_records" in history_data:
                    for item in history_data["raw_records"]:
                        # 复合主键，防止不同任务有同名 ID
                        uid = f"{item['Level']}_{item['Task']}_{item['Image_ID']}"
                        calculated_records[uid] = item
            print(f"📦 发现历史战报！成功加载 {len(calculated_records)} 条已计算数据，准备查漏补缺...")
        except Exception as e:
            print(f"⚠️ 无法读取历史缓存，将重新开始计算: {e}")

    # 2. 初始化两大引擎 (如果您是在类里，保持原样即可)
    # 注意：如果发现全都是存量数据，我们可以连模型都不 load，但这属于深度优化了，目前先这样。
    lpips_eval = LPIPS_Evaluator(device=device)
    dclip_eval = DCLIP_Evaluator(device=device)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    records = []
    new_calculations = 0 # 统计这次新算了多少张

    # 3. 核心评测循环
    for comp, categories in dataset.items():
        print(f"\n 评估层级: [{comp.upper()}]")
        for category, items in categories.items():
            if not items: continue 

            pbar = tqdm(items, desc=f"  任务: {category}", leave=False)
            for item in pbar:
                image_id = item["id"]
                uid = f"{comp.upper()}_{category}_{image_id}"
                source_path = os.path.join(project_root, item["image_path"])

                if uid in calculated_records:
                    # 如果算过，直接把历史分数塞进这次的总池子，然后跳过耗时的计算！
                    records.append(calculated_records[uid])
                    continue
                
                generated_path = os.path.join(generated_dir, comp, category, f"{image_id}.jpg")

                if not os.path.exists(generated_path):
                    continue
                    
                # 指标计算 (这是最耗时的步骤)
                score_lpips = lpips_eval.compute_score(source_path, generated_path)
                score_dclip = dclip_eval.calculate_dclip(
                    src_image_path=source_path, 
                    edit_image_path=generated_path,
                    src_prompt=item.get("source_prompt", ""),
                    tgt_prompt=item.get("target_prompt", ""),
                    instruction=item.get("instruction", "")
                )

                # 把新算出来的成绩也加进总池子 (注意多存了一个 Image_ID)
                records.append({
                    "Image_ID": image_id, 
                    "Method": model_name,
                    "Level": comp.upper(),
                    "Task": category,             
                    "LPIPS (↓)": score_lpips,
                    "D-CLIP (↑)": score_dclip
                })
                new_calculations += 1

    if not records:
        print(f" 错误: 未找到任何有效的生成图片，请检查路径: {generated_dir}")
        return None, None

    if new_calculations > 0:
        print(f"✅ 增量修补完成！本次新计算了 {new_calculations} 张漏网之图。")
    else:
        print(f"🎉 扫描完毕！全部数据已存在，无需重新计算。")

    # 4. 数据聚合处理 (用完整池子 records 重新算平均分，确保最新)
    df = pd.DataFrame(records)
    
    macro_summary = df.groupby("Level")[["LPIPS (↓)", "D-CLIP (↑)"]].mean().round(4).reset_index()
    macro_summary['Count'] = df.groupby("Level").size().values
    
    micro_summary = df.groupby(["Level", "Task"])[["LPIPS (↓)", "D-CLIP (↑)"]].mean().round(4).reset_index()
    micro_summary['Count'] = df.groupby(["Level", "Task"]).size().values
    
    # 5. 【核心】持久化保存为 JSON
    display_name = model_display_name if model_display_name else model_name
    save_data = {
        "model_name": model_name,
        "model_name_display": display_name, 
        "dataset_name": dataset_name,
        "macro_metrics": macro_summary.to_dict(orient='records'),
        "micro_metrics": micro_summary.to_dict(orient='records'),
        "raw_records": records  # 🚨 这里是关键：把每张图的具体分数存入硬盘，留给下次断点续传使用
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n 评测报告已存档/更新: {save_path}")
    print(macro_summary.to_markdown()) 

    return macro_summary, micro_summary
# ==========================================
# 本地测试入口
# ==========================================
if __name__ == "__main__":
    # 测试时请确保路径正确
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = "../../datasets/Hybrid-EditBench-mini.json" # 相对路径
    
    # 示例运行
    run_evaluation(
        model_name="ip2p", 
        dataset_name="mini", 
        dataset_path=test_dataset, 
        model_display_name="InstructPix2Pix"
    )