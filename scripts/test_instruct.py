import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

custom_cache_dir = "/root/autodl-tmp/MyGradProject/models"
model_id = "timbrooks/instruct-pix2pix"

print("正在从本地缓存加载模型...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    safety_checker=None,
    cache_dir=custom_cache_dir,
    local_files_only=True  # 👈 核心修改一：强行加锁！绝对不允许代码去连 Hugging Face！
).to("cuda")


# 假设你在 MyGradProject/inputs/ 下放了一张名为 test.jpg 的测试图
local_image_path = "/root/autodl-tmp/MyGradProject/inputs/test.jpg"  # 👈 核心修改二：把 URL 替换为本地相对路径

print(f"正在读取本地图片: {local_image_path}")
# load_image 这个函数很聪明，传 URL 它就下载，传本地路径它就直接读硬盘
image = load_image(local_image_path) 


prompt = "turn him into a cyborg"

print(f"开始根据指令生成: '{prompt}'...")
# num_inference_steps=20 平衡速度和质量 | image_guidance_scale=1.5 保持原图结构的权重
result_image = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images[0]

# 核心修改三：加入保底机制，防止 FileNotFoundError
output_dir = "/root/autodl-tmp/MyGradProject/outputs"
os.makedirs(output_dir, exist_ok=True)  # 如果 outputs 文件夹不存在，系统会自动帮你建一个

save_path = f"{output_dir}/output_test.png"
result_image.save(save_path)
print(f"生成完成！结果已成功保存至: {save_path}")