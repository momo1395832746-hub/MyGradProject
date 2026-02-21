import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
# 定义新的模型存放路径
custom_cache_dir = "/root/autodl-tmp/MyGradProject/models"

# 定义model_id
model_id = "timbrooks/instruct-pix2pix"

# 加载模型并指定特定加载路径
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    safety_checker=None,
    cache_dir=custom_cache_dir 
).to("cuda")

# 准备测试图片 
url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
image = load_image(url)

# 定义指令
prompt = "turn him into a cyborg"

# 推理生成# num_inference_steps=20 平衡速度和质量# image_guidance_scale=1.5 保持原图结构的权重
print("开始生成...")
image = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images[0]

# 保存结果
image.save("../outputs/output_test.png")
print("生成完成！结果已保存至 /root/autodl-tmp/MyGradProject/outputs/output_test.png")