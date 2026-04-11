import os
from PIL import Image, ImageOps

def standardize_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 智能寻找项目根目录 (向上探索直到发现 datasets 文件夹)
    project_root = current_dir
    while not os.path.exists(os.path.join(project_root, "datasets")):
        parent = os.path.dirname(project_root)
        if parent == project_root: # 已经到达系统根目录
            print("错误: 无法在父级目录中找到 'datasets' 文件夹！")
            return
        project_root = parent
        
    source_dir = os.path.join(project_root, "datasets", "images")
    
    print(f"开始清洗数据集: {source_dir}\n")
    
    count_processed = 0
    count_skipped = 0
    
    for root, dirs, files in os.walk(source_dir):#  os.walk 自动递归遍历所有的 simple 和 complex 子文件夹
        for filename in files:
            # 只处理图片文件
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(root, filename)
            
            # 获取相对路径,看处理的是哪个文件夹里的图片
            rel_path = os.path.relpath(root, source_dir) 
            
            try:
                # 打开图片
                img = Image.open(img_path).convert("RGB")
                
                # 如果不是 512x512，就进行居中裁剪并缩放
                if img.size != (512, 512):
                    print(f" [{rel_path}] 裁剪: {filename} ({img.size} -> 512x512)")
                    img_cleaned = ImageOps.fit(img, (512, 512), Image.Resampling.LANCZOS)
                    
                    # 覆盖原图保存
                    img_cleaned.save(img_path)
                    count_processed += 1
                else:
                    print(f" [{rel_path}] 完美尺寸: {filename} (已是 512x512)")
                    count_skipped += 1
                    
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")

    total = count_processed + count_skipped
    print("\n" + "="*50)
    print(f"双轨制深度清洗完成！共扫描 {total} 张图片。")
    print(f"   -> 实际裁剪修复: {count_processed} 张")
    print(f"   -> 尺寸无需修改: {count_skipped} 张")
    print("="*50 + "\n")

if __name__ == "__main__":
    standardize_dataset()