import torch
import gc
import os
import sys
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline, StableDiffusionPix2PixZeroPipeline, DDIMScheduler, DDIMInverseScheduler
import torch.nn.functional as F

class BaseEditor:
    # 在基类初始化时，统一指定全局缓存目录
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", cache_dir="/root/autodl-tmp/MyGradProject/models"):
        self.device = device
        self.cache_dir = cache_dir  # 全局统一的模型存放地
        self.model = None 
        self.model_name = "Base"

    def load_model(self):
        raise NotImplementedError("子类必须实现 load_model")

    def preprocess_image(self, image: Image.Image, target_size=512) -> Image.Image:
        # 统一预处理：等比例缩放并居中裁剪至 512x512，保证输入基准一致
        if image.mode != "RGB":
            image = image.convert("RGB")
        return ImageOps.fit(image, (target_size, target_size), Image.Resampling.LANCZOS)

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "", seed: int = 42, **kwargs) -> Image.Image:
        """
        图像编辑的统一接口
        :param image: 输入的原图
        :param prompt: 目标提示词 (Target Prompt) 或 编辑指令 (Instruction)
        :param source_prompt: 源提示词 (Source Prompt) 
        :param seed: 随机种子，默认 42，用于锁死生成方差，保证实验可复现
        :param kwargs: 其他高级控制参数 (如 steps, cfg_scale 等)
        """
        raise NotImplementedError("子类必须实现 edit_image")
        
    def clear_vram(self):
        # 显存清理逻辑，防止切换模型时 OOM
        print(f"[{self.model_name}] 清理显存中...")
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class SDEditEditor(BaseEditor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "SDEdit"
        self.model_id = "runwayml/stable-diffusion-v1-5" 

    def load_model(self):
        from diffusers import StableDiffusionImg2ImgPipeline
        print(f"[{self.model_name}] 正在加载...")
        self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir, 
            torch_dtype=torch.float16, 
            safety_checker=None,
            local_files_only=True
        ).to(self.device)

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "",seed: int = 42, **kwargs) -> Image.Image:
        if self.model is None: self.load_model()
        
        steps = kwargs.get("num_inference_steps", 50)
        txt_cfg = kwargs.get("guidance_scale", 7.5)
        
        print(f"[{self.model_name}] 执行参数 -> Steps: {steps}, TextCFG: {txt_cfg}, Seed: {seed}")
        processed_image = self.preprocess_image(image)

        #创建一个固定种子的生成器
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.model(
            prompt=prompt,
            image=processed_image,
            strength=0.7, #strength 决定加噪程度 (0.7表示加70%的噪声)
            num_inference_steps=steps,
            guidance_scale=txt_cfg,
            generator=generator
        ).images[0]
        
        return result

class InstructPix2PixEditor(BaseEditor):
    # 子类使用 **kwargs 接收所有未显式声明的关键字参数
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "InstructPix2Pix"
        self.model_id = "timbrooks/instruct-pix2pix" 

    def load_model(self):
        print(f" 正在从本地加载模型: {self.model_name} ...")
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16, 
            safety_checker=None,
            local_files_only=True
        ).to(self.device)
        print(" 模型加载并已转移至 GPU！")

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "",seed: int = 42, **kwargs) -> Image.Image:
        if self.model is None:
            self.load_model()
        else:
            # 如果模型之前被踢到了 CPU，现在把它拉回 GPU
            self.model.to(self.device)
            
        processed_image = self.preprocess_image(image)

        # 从 kwargs 安全提取滑块传来的参数，赋予默认兜底值
        steps = kwargs.get("num_inference_steps", 20)
        img_cfg = kwargs.get("image_guidance_scale", 1.5)
        txt_cfg = kwargs.get("guidance_scale", 7.5)

        print(f"[{self.model_name}] 执行参数 -> Steps: {steps}, TextCFG: {txt_cfg}, ImageCFG: {img_cfg}, Seed: {seed}")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 使用动态参数进行推理
        result = self.model(
            prompt=prompt, 
            image=processed_image, 
            num_inference_steps=steps,          
            image_guidance_scale=img_cfg,        
            guidance_scale=txt_cfg,
            generator=generator
        ).images[0]
        

        print(f"[{self.model_name}] 图片编辑完成...")
        print("=" * 60 + "\n\n")
        
        return result

class Pix2PixZeroEditor(BaseEditor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Pix2Pix-Zero"
        self.model_id = "runwayml/stable-diffusion-v1-5" 

    def load_model(self):
        from diffusers import StableDiffusionPix2PixZeroPipeline, DDIMScheduler, DDIMInverseScheduler
        print(f"[{self.model_name}] 正在加载官方 Pipeline...")
        self.model = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir, 
            torch_dtype=torch.float16, 
            safety_checker=None,
            local_files_only=True
        ).to(self.device)
        self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)
        self.model.inverse_scheduler = DDIMInverseScheduler.from_config(self.model.scheduler.config)

    def get_paper_equivalent_embeds(self, prompt_text):
        # Prompt Ensembling (提示词集成)，使用 20+ 个模板，模拟 GPT-3 的多样性输出(后续可采取模型真实接入进行扩展)
        templates = [
            "a photo of a {}", "a rendering of a {}", "a cropped photo of the {}",
            "the photo of a {}", "a photo of a clean {}", "a photo of a dirty {}",
            "a dark photo of the {}", "a picture of a {}", "a cool photo of a {}",
            "a close-up photo of a {}", "a bright photo of the {}", "a cropped photo of a {}",
            "a photo of the {}", "a good photo of the {}", "a photo of one {}",
            "a close-up photo of the {}", "a rendition of the {}", "a photo of the clean {}",
            "a rendition of a {}", "a photo of a nice {}", "a good photo of a {}"
        ]
        prompts = [t.format(prompt_text) for t in templates]
        
        # 严格调用底层 Tokenizer 和 Text Encoder 提取特征
        text_inputs = self.model.tokenizer(
            prompts, padding="max_length", max_length=self.model.tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.model.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # 沿着 batch 维度求平均，获取语义方向
        return text_embeddings.mean(dim=0, keepdim=True)

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "", seed: int = 42, **kwargs) -> Image.Image:
        if self.model is None: self.load_model()
        
        steps = kwargs.get("num_inference_steps", 50)
        cag_amount = kwargs.get("cross_attention_guidance_amount", 0.1) 
        txt_cfg = kwargs.get("guidance_scale", 5.0) 
        
        print(f"[{self.model_name}] 执行参数 -> Steps: {steps}, CAG: {cag_amount}, TextCFG: {txt_cfg}, Seed: {seed}")
        processed_image = self.preprocess_image(image)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        print(f"[{self.model_name}] 1. 执行DDIM 反演 (锁定 CFG=1.0)...")#原方法采取的反演参数

        inv_latents = self.model.invert(
            source_prompt, 
            image=processed_image, 
            guidance_scale=1.0,  
            generator=generator, 
            num_inference_steps=steps
        ).latents
        
        print(f"[{self.model_name}] 2. 计算集成语义方向...")
        # 调用大批量特征求均值函数
        source_embeds = self.get_paper_equivalent_embeds(source_prompt)
        target_embeds = self.get_paper_equivalent_embeds(prompt)
        
        print(f"[{self.model_name}] 3. 执行交叉注意力引导生成...")
        result = self.model(
            prompt=prompt,
            source_embeds=source_embeds,
            target_embeds=target_embeds,
            latents=inv_latents,
            guidance_scale=txt_cfg,
            cross_attention_guidance_amount=cag_amount,
            num_inference_steps=steps,
            generator=generator
        ).images[0]
        
        print(f"[{self.model_name}] 4. 图片编辑完成...")
        print("=" * 60 + "\n\n")
        return result