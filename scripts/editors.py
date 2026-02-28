import torch
import gc
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline, DDIMScheduler
from ptp_utils import AttentionStore, AttentionControlEdit, register_attention_control
import torch.nn.functional as F

class BaseEditor:
    # 1. 在基类初始化时，统一指定全局缓存目录
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

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "", **kwargs) -> Image.Image:
        """
        图像编辑的统一接口
        :param image: 输入的原图
        :param prompt: 目标提示词 (Target Prompt) 或 编辑指令 (Instruction)
        :param source_prompt: 源提示词 (Source Prompt) - P2P/NTI 算法必需
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
        self.model_name = "SDEdit (加噪去噪基准)"
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

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "", **kwargs) -> Image.Image:
        if self.model is None: self.load_model()
        
        steps = kwargs.get("num_inference_steps", 50)
        txt_cfg = kwargs.get("guidance_scale", 7.5)
        
        print(f"[{self.model_name}] 执行参数 -> Steps: {steps}, TextCFG: {txt_cfg}")
        processed_image = self.preprocess_image(image)
        
        # SDEdit 的核心：strength 决定加噪程度 (0.7表示加70%的噪声)
        result = self.model(
            prompt=prompt,
            image=processed_image,
            strength=0.7, 
            num_inference_steps=steps,
            guidance_scale=txt_cfg
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

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "", **kwargs) -> Image.Image:
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

        print(f"[{self.model_name}] 执行参数 -> Steps: {steps}, TextCFG: {txt_cfg}, ImageCFG: {img_cfg}")
        
        # 使用动态参数进行推理
        result = self.model(
            prompt=prompt, 
            image=processed_image, 
            num_inference_steps=steps,          
            image_guidance_scale=img_cfg,        
            guidance_scale=txt_cfg               
        ).images[0]
        

        print(f"[{self.model_name}] 图片编辑完成...")
        print("=" * 60 + "\n\n")
        
        return result

class P2P_NTI_Editor(BaseEditor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Prompt-to-Prompt w/ NTI "
        self.model_id = "runwayml/stable-diffusion-v1-5" 

    def load_model(self):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        print(f"[{self.model_name}] 正在加载...")
        self.model = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir, 
            torch_dtype=torch.float16, 
            safety_checker=None,
            local_files_only=True
        ).to(self.device)
        # 🚨 终极修复 1：强行关闭截断！保证正反向数学轨迹 100% 对称！
        self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config, clip_sample=False)
        # 🚨 新增：强制把 VAE 转为 float32 防爆！
        self.model.vae.to(dtype=torch.float32)

    def edit_image(self, image: Image.Image, prompt: str, source_prompt: str = "", **kwargs) -> Image.Image:
        steps = kwargs.get("num_inference_steps", 50)
        txt_cfg = kwargs.get("guidance_scale", 7.5)
        cross_replace = kwargs.get("cross_replace_steps", 0.8)
        # 👈 新增：安全提取前端传来的自注意力替换参数
        self_replace = kwargs.get("self_replace_steps", 0.5)
        inner_steps = kwargs.get("num_inner_steps", 10)

        print(f"[{self.model_name}] 执行参数 -> Steps: {steps}, TextCFG: {txt_cfg}, CrossReplace: {cross_replace}, InnerSteps: {inner_steps}")
        
        if self.model is None: self.load_model()
            
        print(f"[{self.model_name}] 1. 提取原图潜变量...")
        processed_image = self.preprocess_image(image)
        image_tensor = self.model.image_processor.preprocess(processed_image).to(self.device, dtype=torch.float)
        with torch.no_grad():
            latent_original = self.model.vae.encode(image_tensor).latent_dist.sample() * self.model.vae.config.scaling_factor
            latent_original = latent_original.to(torch.float16)
        # 文本编码
        text_inputs = self.model.tokenizer([source_prompt, prompt], padding="max_length", max_length=self.model.tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = self.model.text_encoder(text_inputs.input_ids.to(self.device))[0]
        uncond_input = self.model.tokenizer(["", ""], padding="max_length", max_length=self.model.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # ==========================================
        # 🚨 核心修复：把 Batch 维度精准切分为单份！
        # ==========================================
        cond_inv = text_embeddings[0:1]    # 源提示词特征 [1, 77, 768]
        cond_tgt = text_embeddings[1:2]    # 目标提示词特征 [1, 77, 768]
        uncond_inv = uncond_embeddings[0:1]# 源空提示词特征 [1, 77, 768]

        print(f"[{self.model_name}] 2. 获取 DDIM 轨迹...")
        self.model.scheduler.set_timesteps(steps)
        inv_timesteps = reversed(self.model.scheduler.timesteps)
        latents_inv = latent_original.clone()
        
        latent_trajectory = [latents_inv]
        step_ratio = self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        
        with torch.no_grad():
            for t in inv_timesteps:
                t_next = t + step_ratio
                noise_pred = self.model.unet(latents_inv, t, encoder_hidden_states=cond_inv).sample
                
                alpha_prod_t = self.model.scheduler.alphas_cumprod[t]
                if t_next < self.model.scheduler.config.num_train_timesteps:
                    alpha_prod_t_next = self.model.scheduler.alphas_cumprod[t_next]
                else:
                    alpha_prod_t_next = torch.tensor(0.0, dtype=latents_inv.dtype, device=self.device)
                
                pred_x0 = (latents_inv - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                latents_inv = alpha_prod_t_next ** 0.5 * pred_x0 + (1 - alpha_prod_t_next) ** 0.5 * noise_pred
                latent_trajectory.append(latents_inv)

        print(f"[{self.model_name}] 3. 执行 Null-text 优化循环...")
        uncond_embeddings_list = []
        latent_trajectory = latent_trajectory[::-1]
        
        timesteps = self.model.scheduler.timesteps
        for i, t in enumerate(timesteps):
            # 🚨 终极修复 2：将优化的参数强行拉升到 float32 单精度！
            uncond_opt = uncond_inv.clone().detach().to(torch.float32).requires_grad_(True)
            optimizer = torch.optim.Adam([uncond_opt], lr=1e-2)
            
            latent_prev = latent_trajectory[i].detach()
            latent_target = latent_trajectory[i+1].detach() if i+1 < len(latent_trajectory) else latent_original.detach()
            
            with torch.no_grad():
                noise_pred_cond = self.model.unet(latent_prev, t, encoder_hidden_states=cond_inv).sample
            
            for _ in range(inner_steps):
                optimizer.zero_grad()
                
                # U-Net 必须吃 float16 的数据，所以这里临时降维传进去
                noise_pred_uncond = self.model.unet(latent_prev, t, encoder_hidden_states=uncond_opt.to(torch.float16)).sample
                
                noise_pred = noise_pred_uncond + txt_cfg * (noise_pred_cond - noise_pred_uncond)
                latent_pred = self.model.scheduler.step(noise_pred, t, latent_prev).prev_sample
                
                # 🚨 终极修复 3：MSE 损失计算必须在 float32 下进行，杜绝下溢出导致 NaN
                loss = torch.nn.functional.mse_loss(latent_pred.to(torch.float32), latent_target.to(torch.float32))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_([uncond_opt], max_norm=1.0)
                optimizer.step()
                
            # 优化完一轮，降维成 float16 存起来，供最终生成步骤使用
            uncond_embeddings_list.append(uncond_opt.detach().to(torch.float16))
            if (i+1) % 5 == 0:
                print(f"  -> 优化进度: {i+1}/{steps} 步 (Loss: {loss.item():.4f})")

        print(f"[{self.model_name}] 4. 开始 P2P 生成 (使用优化后的 Null-text)...")
        latents = torch.cat([latent_trajectory[0]] * 2) 
        # 👈 核心修改：使用支持双重控制的新控制器 AttentionControlEdit
        controller = AttentionControlEdit(
            prompts=[source_prompt, prompt], 
            num_steps=steps, 
            cross_replace_steps=cross_replace,
            self_replace_steps=self_replace # 传入自注意力阈值
        )
        original_attn_procs = self.model.unet.attn_processors
        register_attention_control(self.model.unet, controller)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
            
            # ==========================================
            # 🚨 重构拼接维度：严格保证喂给 U-Net 的 Batch Size 为 4！
            # ==========================================
            step_uncond = uncond_embeddings_list[i] # 这是优化好的 [1, 77, 768]
            step_text_embeddings = torch.cat([
                step_uncond,       # 给源图用的空特征 (1)
                step_uncond,       # 给目标图用的空特征 (1)
                cond_inv,          # 源提示词特征 (1)
                cond_tgt           # 目标提示词特征 (1)
            ])                     # 拼合后完美等于 [4, 77, 768]，不崩！
            
            with torch.no_grad():
                noise_pred = self.model.unet(latent_model_input, t, encoder_hidden_states=step_text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + txt_cfg * (noise_pred_text - noise_pred_uncond)
            latents = self.model.scheduler.step(noise_pred, t, latents).prev_sample
            latents = controller.step_callback(latents)

        print(f"[{self.model_name}] 5. 潜变量解码中...")
        with torch.no_grad():
            # 1. 提拉回 float32 给 VAE
            latents_f32 = latents.to(torch.float32) / self.model.vae.config.scaling_factor
            # 2. 安全解码
            image_out = self.model.vae.decode(latents_f32, return_dict=False)[0]
            image_out = self.model.image_processor.postprocess(image_out, output_type="pil")
        
        self.model.unet.set_attn_processor(original_attn_procs)

        print(f"[{self.model_name}] 6. 图片编辑完成...")
        print("=" * 60 + "\n\n")
        return image_out[1]

