import json
import os

def build_dataset_template():
    # 1. 定义存储路径 (基于你项目根目录)
    dataset_dir = "data"
    images_dir = os.path.join(dataset_dir, "images")
    output_json_path = os.path.join(dataset_dir, "Hybrid-EditBench.json")

    # 2. 自动创建文件夹
    os.makedirs(images_dir, exist_ok=True)

    # 3. 满血版 5大类 种子数据
    seed_data = {
        # 类别 1: 刚性对象替换
        "object_replacement": [
            {
                "id": "O_001",
                "image_path": f"{images_dir}/{object_replacement}/O_001.jpg",
                "source_prompt": "A table setting on a red carpet in a living room",
                "target_prompt": "A dog setting on a red carpet in a living room",
                "instruction": "change the table for a dog",
                "dataset_source": "MagicBrush"
            }

            {
                "id": "O_002",
                "image_path": f"{images_dir}/{object_replacement}/O_002.jpg",
                "source_prompt": "A car trunk full of colorful bags and a yellow suitcase covered by a black cargo net",
                "target_prompt": "A car trunk full of colorful bags and a yellow suitcase",
                "instruction": "remove the mesh screen",
                "dataset_source": "MagicBrush"
            }

            {
                "id": "O_003",
                "image_path": f"{images_dir}/{object_replacement}/O_003.jpg",
                "source_prompt": "Two blue umbrellas on a green grassy lawn with trees in the background",
                "target_prompt": "A big dog and two blue umbrellas on a green grassy lawn with trees in the background",
                "instruction": "Add a big dog.",
                "dataset_source": "MagicBrush"
            }
        ],
        
        # 类别 2: 局部属性修改
        "attribute_modification": [
            {
                "id": "A_001",
                "image_path": f"{images_dir}/{attribute_modification}/A_001.jpg",
                "source_prompt": "A woman wearing a blue tank top making a pizza on a kitchen counter",
                "target_prompt": "A woman wearing a red tank top making a pizza on a kitchen counter",
                "instruction": "change the blue shirt to red shirt",
                "dataset_source": "MagicBrush"
            }

            {
                "id": "A_002",
                "image_path": f"{images_dir}/{attribute_modification}/A_002.jpg",
                "source_prompt": "Several wooden baseball bats resting on some papers and pens on a table",
                "target_prompt": "Several steel baseball bats resting on some papers and pens on a table",
                "instruction": "let it be steel bats",
                "dataset_source": "MagicBrush"
            }
        ],

        # 类别 3: 非刚性动作/形变
        "non_rigid_action": [
            {
                "id": "N_001",
                "image_path": f"{images_dir}/{non_rigid_action}/N_001.png",
                "source_prompt": "A photo of a blue and orange bird resting on a wooden branch",
                "target_prompt": "A photo of a blue and orange bird spreading its wings on a wooden branch",
                "instruction": "Make the bird spread its wings",
                "dataset_source": "TEDBench"
            }

            {
                "id": "N_002",
                "image_path": f"{images_dir}/{non_rigid_action}/N_002.png",
                "source_prompt": "A photo of a closed brown wooden door on a dark grey wall",
                "target_prompt": "A photo of an open brown wooden door on a dark grey wall",
                "instruction": "Open the door",
                "dataset_source": "TEDBench"
            }
        ],

        # 类别 4: 全局/风格迁移
        "style_transfer": [
            {
                "id": "S_001",
                "image_path": f"{images_dir}/{style_transfer}/S_001.jpg",
                "source_prompt": "A realistic photo of a deer behind a campfire in a snowy pine forest",
                "target_prompt": "A cartoon illustration of a deer behind a campfire in a snowy pine forest",
                "instruction": "Make it a cartoon",
                "dataset_source": "IP2P"
            }

            {
                "id": "S_002",
                "image_path": f"{images_dir}/{style_transfer}/S_002.jpg",
                "source_prompt": "A beautiful landscape of snow-capped mountains and red hills at sunset",
                "target_prompt": "A beautiful landscape of snow-capped mountains and red hills at sunrise",
                "instruction": "change the sunset to a sunrise",
                "dataset_source": "IP2P"
            }
        ],

        # 类别 5: 复杂语义/推理编辑
        "reasoning": [
            {
                "id": "R_001",
                "image_path": f"{images_dir}/{reasoning}/R_001.png",
                "source_prompt": "A man wearing a casual t-shirt",
                "target_prompt": "A man going to a formal funeral",
                "instruction": "Make him look like he is going to a formal funeral",
                "dataset_source": "Internet"
            }
        ]
    }

    # 4. 写入 JSON 文件
    print("正在生成满血版 (5大类) Hybrid-EditBench 数据集骨架...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        # indent=4 保证 JSON 文件格式美观，方便你肉眼检查和手工修改
        json.dump(seed_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 成功构建 5 大类嵌套分类结构！")
    print(f"📂 配置文件已保存至: {output_json_path}")
    print(f"🖼️ 请去网上或数据集中找对应的图，重命名后放入: {images_dir}/ 目录下")
    print(f"📝 接下来请直接打开 JSON 文件，在每个大类的列表 [] 中，复制 {{...}} 块来扩充数据，凑齐 10 张图。")

if __name__ == "__main__":
    build_dataset_template()