import os
import gradio as gr

# 1. 定义数据盘路径
DATA_DISK_PATH = "/root/autodl-tmp/hf_cache"

# 2. 物理创建文件夹（防止因为文件夹不存在而滑向默认路径）
if not os.path.exists(DATA_DISK_PATH):
    os.makedirs(DATA_DISK_PATH, exist_ok=True)
    print(f"✅ 已物理创建数据盘缓存区: {DATA_DISK_PATH}")

# 3. 强制注入环境变量（必须在 import datasets 之前！）
os.environ["HF_DATASETS_CACHE"] = DATA_DISK_PATH
os.environ["HF_HOME"] = DATA_DISK_PATH # 这一行是双重保险
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

def fetch_images(dataset_id, split_name, start_idx, num_images):
    """后台核心逻辑：硬核物理切片下载，拒绝网络闪断"""
    try:
        start = int(start_idx)
        count = int(num_images)
        
        # 🎯 核心修改：锁定第一个分片。
        # IP2P 的文件结构通常是 data/train-00000-of-00262-xxx.parquet
        # 如果是其他数据集，可能需要调整 data_files 的通配符
        target_shard = "data/train-00000-of-*" 
        
        print(f"\n[📦 单包作战] 目标数据集: {dataset_id}")
        print(f"[📦 单包作战] 锁定分片: {target_shard}")

        # 强制只加载命中的文件，忽略其余 261 个分片
        dataset = load_dataset(
            dataset_id, 
            data_files={split_name: target_shard}, 
            split=split_name,    
            verification_mode="no_checks"
        )
        
        images = []
        # 计算该分片内的有效范围
        end_idx = min(start + count, len(dataset))
        
        print(f"[后台行动] 正在读取分片内索引: {start} 到 {end_idx}")
        
        for i in range(start, end_idx):
            item = dataset[i]
            img = item.get('original_image') or item.get('image_0') or item.get('source_img')
            if img:
                # 标记分片内的全局编号
                images.append((img, f"编号: {i}"))
                
        print(f"[后台捷报] 成功从首个分片提取 {len(images)} 张图片！")
        return images
    except Exception as e:
        print(f"[报错] {str(e)}")
        # 如果报错 401，说明需要去 HF 官网生成 Token 并填入
        raise gr.Error(f"抓取失败: {str(e)}")

# 构建前端雷达面板
with gr.Blocks(title="HF 单包雷达站 (AutoDL版)", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("## 🎯 HF 终极雷达站 - 单包提取模式")
    gr.Markdown("ℹ️ **策略**：此版本强制只下载数据集的第 1 个分片（约 500MB），彻底跳过其余 130GB 冗余数据。")
    
    with gr.Row():
        d_id = gr.Textbox(label="数据集名称", value="timbrooks/instructpix2pix-clip-filtered", scale=2)
        d_split = gr.Textbox(label="Split 名称", value="train", scale=1)
    
    with gr.Row():
        d_start = gr.Number(label="起始索引 (0-1500)", value=0, precision=0, scale=1)
        d_num = gr.Number(label="抓取数量", value=40, precision=0, scale=1)
        btn = gr.Button("🚀 局部扫描", variant="primary", scale=1)
    
    gallery = gr.Gallery(
        label="作战影像库", show_label=False, columns=[4], rows=[2], object_fit="contain", height="auto"
    )
    
    # 修改了 inputs，加入了 d_start
    btn.click(fn=fetch_images, inputs=[d_id, d_split, d_start, d_num], outputs=gallery)

if __name__ == "__main__":
    print("✅ 单包雷达站准备升空！")
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)