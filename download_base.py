import os
import torch
from diffusers import StableDiffusionPipeline

# 1. 强制启用国内加速镜像（极其重要，防止下载卡死！）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 指定你的统一模型车库
cache_dir = "/root/autodl-tmp/MyGradProject/models"
model_id = "runwayml/stable-diffusion-v1-5"

print(f"开始从 HF 镜像站下载 {model_id} ...")
print(f"保存路径: {cache_dir}")
print("注意：模型大约 4-5 GB，视网速可能需要 3-10 分钟，请耐心等待进度条...")

# 3. 执行下载（只下载，不加载到显卡以节省资源）
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    cache_dir=cache_dir,
    torch_dtype=torch.float16, # 使用半精度，既省硬盘又省后续的显存
    safety_checker=None
)

print("🎉 底座模型下载完成！你的 '原厂赛车' 已经入库！")