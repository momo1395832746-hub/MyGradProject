import torch
import math
import abc
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        self.cur_step += 1
        self.cur_att_layer = 0
        return x_t

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer == 0:
            self.num_att_layers = 0
        self.num_att_layers += 1
        self.cur_att_layer += 1
        return self.forward(attn, is_cross, place_in_unet)

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        pass

class AttentionStore(AttentionControl):
    # 基础的注意力存储器，负责把 SD 里的 Attention Map 偷出来存进字典
    def __init__(self):
        super().__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def get_empty_store(self):
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # 存储当前的注意力图
        self.step_store[key].append(attn)
        return attn

    def step_callback(self, x_t):
        # 每走完一个 timestep，就把这一步的注意力图保存下来并清空临时仓库
        super().step_callback(x_t)
        for key in self.step_store:
            if key not in self.attention_store:
                self.attention_store[key] = []
            self.attention_store[key].extend(self.step_store[key])
        self.step_store = self.get_empty_store()
        return x_t
        
    def reset(self):
        super().__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

import math
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

# ==========================================
# 1. 核心控制器：支持自注意力与跨注意力替换 (修复了缺失的 self_replace)
# ==========================================
class AttentionControlEdit(AttentionControl):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float):
        super().__init__()
        self.batch_size = len(prompts)
        self.num_steps = num_steps
        self.cross_replace_alpha = cross_replace_steps 
        self.self_replace_alpha = self_replace_steps # 👈 新增：自注意力替换阈值

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        
        # 应对 CFG 带来的 4 倍 Batch 维度 [无条件源, 无条件目标, 有条件源, 有条件目标]
        chunk_size = attn.shape[0] // 4
        
        if is_cross:
            # 【跨注意力替换】：保证语义位置对应（如目标图像的“狗”生成在原图“猫”的位置）
            if self.cur_step < int(self.num_steps * self.cross_replace_alpha):
                attn[chunk_size : 2 * chunk_size] = attn[0 : chunk_size].clone()
                attn[3 * chunk_size : 4 * chunk_size] = attn[2 * chunk_size : 3 * chunk_size].clone()
        else:
            # 【自注意力替换】：保证几何结构与背景纹理不变
            if self.cur_step < int(self.num_steps * self.self_replace_alpha):
                attn[chunk_size : 2 * chunk_size] = attn[0 : chunk_size].clone()
                attn[3 * chunk_size : 4 * chunk_size] = attn[2 * chunk_size : 3 * chunk_size].clone()
            
        return attn

# ==========================================
# 2. U-Net 劫持钩子 (Monkey Patch)
# ==========================================
class P2PAttnProcessor:
    def __init__(self, controller, place_in_unet):
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        is_cross = encoder_hidden_states is not None
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query, key.transpose(-1, -2), beta=0, alpha=1.0 / math.sqrt(query.shape[-1])
        )
        attention_probs = attention_scores.softmax(dim=-1)
        
        # 拦截并篡改注意力矩阵
        attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_control(unet, controller):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        place_in_unet = "down" if "down" in name else "up" if "up" in name else "mid"
        attn_procs[name] = P2PAttnProcessor(controller, place_in_unet)
    unet.set_attn_processor(attn_procs)