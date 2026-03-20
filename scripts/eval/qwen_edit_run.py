import os
import json
import time
import requests
import dashscope
from PIL import Image
from tqdm import tqdm
import argparse  

# 1. 全局配置与key检测
if "DASHSCOPE_API_KEY" not in os.environ:
    print(" 错误：未检测到 DASHSCOPE_API_KEY 环境变量！")
    exit(1)

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 2. Qwen-Image-Edit 
def call_qwen_edit_api(source_path, prompt, max_retries=3):
    attempt = 0
    local_image_uri = f"file://{source_path}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": local_image_uri}, 
                {"text": prompt}            
            ]
        }
    ]

    while attempt < max_retries:
        try:
            response = dashscope.MultiModalConversation.call(
                model="qwen-image-edit-plus",
                messages=messages,
                result_format='message',
                stream=False,
                n=1,               
                watermark=False,    
                seed=42
            )
            
            if response.status_code == 200:
                content_list = response.output.choices[0].message.content
                image_url = None
                for item in content_list:
                    if 'image' in item:
                        image_url = item['image']
                        break
                
                if image_url:
                    img_response = requests.get(image_url, timeout=15)
                    img_response.raise_for_status()
                    return img_response.content
                else:
                    raise Exception("API 返回成功，但未在JSON里找到图片链接")
            else:
                raise Exception(f"百炼 API 报错: {response.code} - {response.message}")

        except Exception as e:
            attempt += 1
            wait_time = 2 ** attempt
            tqdm.write(f"！！第 {attempt} 次调用或下载失败 ({e})，等待 {wait_time}s...")
            time.sleep(wait_time)

    tqdm.write("重试已达上限，放弃该图片")
    return None

# 3. 主循环 
def run_batch_evaluation(dataset_path):
    print(" 启动 [阿里百炼 Qwen-Image-Edit-Plus] 自动跑图程序...")

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    # 统一输出目录：严格对齐 ip2p 的逻辑，直接叫 'qwen'
    output_dir = os.path.join(project_root, "datasets", "generated", "qwen")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n 开始编辑图像 (使用测试集: {os.path.basename(dataset_path)})\n")
    
    for comp, categories in dataset.items():
        print(f"\n======== 当前评测维度: [{comp.upper()}] ========")
        for category, items in categories.items():
            
            # 自动构建带复杂度的多级输出目录
            cat_out_dir = os.path.join(output_dir, comp, category)
            os.makedirs(cat_out_dir, exist_ok=True)
            
            for item in tqdm(items, desc=f"  正在生成: {category}"):
                image_id = item["id"]
                source_path = os.path.join(project_root, item["image_path"])
                out_path = os.path.join(cat_out_dir, f"{image_id}.jpg")
                
                # 断点续传检查
                if os.path.exists(out_path):
                    try:
                        Image.open(out_path).verify()
                        continue
                    except Exception:
                        tqdm.write(f"发现损坏图片 {image_id}，准备重试...")

                if not os.path.exists(source_path):
                    tqdm.write(f"   找不到原图 {source_path}，跳过！")
                    continue

                # 调用云端 API
                img_data = call_qwen_edit_api(
                    source_path=source_path, 
                    prompt=item["instruction"] 
                )

                if img_data:
                    with open(out_path, 'wb') as f:
                        f.write(img_data)
                
    print("\n Qwen-Edit 所有任务生成完毕！")
    print(f" 请前往 {output_dir} 查看结果。")

if __name__ == "__main__":
    # 严格对齐 ip2p_run.py 的参数接收方式
    parser = argparse.ArgumentParser(description="批量运行图像编辑 (云端 API)")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 JSON 文件的绝对或相对路径")
    args = parser.parse_args()
    
    run_batch_evaluation(args.dataset)