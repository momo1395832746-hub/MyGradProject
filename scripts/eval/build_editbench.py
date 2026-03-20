import json
import os

def build_dataset_template():
    # 路径定义
    current_file_path = os.path.abspath(__file__)
    scripts_eval_dir = os.path.dirname(current_file_path)
    scripts_dir = os.path.dirname(scripts_eval_dir)
    project_root = os.path.dirname(scripts_dir)
    
    dataset_dir = os.path.join(project_root, "datasets")
    images_dir = os.path.join(dataset_dir, "images")
    output_json_path = os.path.join(dataset_dir, "Hybrid-EditBench.json")


    # 图像分简单与复杂任务分为五类任务
    seed_data = {
        "complex": {
            "object_replacement": [
                {
                    "id": "O_001",
                    "image_path": "datasets/images/complex/object_replacement/O_001.jpg",
                    "source_prompt": "A table setting on a red carpet in a living room",
                    "target_prompt": "A dog setting on a red carpet in a living room",
                    "instruction": "change the table for a dog",
                    "dataset_source": "MagicBrush"
                },
                {
                    "id": "O_002",
                    "image_path": "datasets/images/complex/object_replacement/O_002.jpg",
                    "source_prompt": "A car trunk full of colorful bags and a yellow suitcase covered by a black cargo net",
                    "target_prompt": "A car trunk full of colorful bags and a yellow suitcase",
                    "instruction": "remove the mesh screen",
                    "dataset_source": "MagicBrush"
                },
                {
                    "id": "O_003",
                    "image_path": "datasets/images/complex/object_replacement/O_003.jpg",
                    "source_prompt": "A scenic rolling green landscape with scattered pine trees, a dirt path, and a blue sky",
                    "target_prompt": "A scenic rolling green landscape with scattered pine trees, a dirt path, a blue sky, and a dog on the right",
                    "instruction": "Add a dog to the right side of the grass",
                    "dataset_source": "MagicBrush"
                }
            ],
            
            "attribute_modification": [
                {
                    "id": "A_001",
                    "image_path": "datasets/images/complex/attribute_modification/A_001.jpg",
                    "source_prompt": "A woman wearing a blue tank top making a pizza on a kitchen counter",
                    "target_prompt": "A woman wearing a red tank top making a pizza on a kitchen counter",
                    "instruction": "change the tank top to red",
                    "dataset_source": "MagicBrush"
                },
                {
                    "id": "A_002",
                    "image_path": "datasets/images/complex/attribute_modification/A_002.jpg",
                    "source_prompt": "A group of wooden baseball bats resting on some papers and pens on a table",
                    "target_prompt": "A group of steel baseball bats resting on some papers and pens on a table",
                    "instruction": "Make the baseball bats steel",
                    "dataset_source": "MagicBrush"
                }
            ],
            
            "non_rigid_action": [
                {
                    "id": "N_001",
                    "image_path": "datasets/images/complex/non_rigid_action/N_001.jpg",
                    "source_prompt": "A black dog and a yellow dog on a grassy field, both with resting expressions.",
                    "target_prompt": "A black dog and a yellow dog on a grassy field, both with panting expressions.",
                    "instruction": "Make the dogs pant",
                    "dataset_source": "MagicBrush"
                },
                {
                    "id": "N_002",
                    "image_path": "datasets/images/complex/non_rigid_action/N_002.jpg",
                    "source_prompt": "A photograph of an elephant walking on a dusty savanna field with its trunk lowered and slightly curled inward.",
                    "target_prompt": "A photograph of an elephant walking on a dusty savanna field with its trunk raised high and trumpeting in the air.",
                    "instruction": "Make the elephant raise its trunk",
                    "dataset_source": "MagicBrush"
                }
            ], 
            
            "style_transfer": [
                {
                    "id": "S_001",
                    "image_path": "datasets/images/complex/style_transfer/S_001.jpg",
                    "source_prompt": "A realistic photo of a deer behind a campfire in a snowy pine forest",
                    "target_prompt": "A cartoon illustration of a deer behind a campfire in a snowy pine forest",
                    "instruction": "Make it a cartoon",
                    "dataset_source": "IP2P"
                },
                {
                    "id": "S_002",
                    "image_path": "datasets/images/complex/style_transfer/S_002.jpg",
                    "source_prompt": "A lush landscape with a dense green forest, a blue pond, grass, and rocks, flat digital illustration style.",
                    "target_prompt": "A lush landscape with a dense green forest, a blue pond, grass, and rocks, masterpiece oil painting style.",
                    "instruction": "turn into an oil painting",
                    "dataset_source": "IP2P"
                }
            ],  
            
            "reasoning": [
                {
                    "id": "R_001",
                    "image_path": "datasets/images/complex/reasoning/R_001.jpg",
                    "source_prompt": "A man wearing a casual t-shirt",
                    "target_prompt": "A man going to a formal funeral",
                    "instruction": "Make him look like he is going to a formal funeral",
                    "dataset_source": "Internet"
                }
            ]
        },

        
        "simple": {
            "object_replacement": [
                {
                    "id": "O_001",
                    "image_path": "datasets/images/simple/object_replacement/O_001.jpg",
                    "source_prompt": "A basket of apples placed in front of white cloth",
                    "target_prompt": "A bowl of apples placed in front of white cloth",
                    "instruction": "Change the basket to a bowl",
                    "dataset_source": "TEDBench"
                },
                {
                    "id": "O_002",
                    "image_path": "datasets/images/simple/object_replacement/O_002.jpg",
                    "source_prompt": "A banana on a wooden desktop",
                    "target_prompt": "Two bananas on a wooden desktop",
                    "instruction": "Add a banana.",
                    "dataset_source": "TEDBench"
                },
                {
                    "id": "O_003",
                    "image_path": "datasets/images/simple/object_replacement/O_003.jpg",
                    "source_prompt": "A pizza with peperoni in front of some bell pepers",
                    "target_prompt": "A pizza without peperoni in front of some bell pepers",
                    "instruction": "Remove the pepperoni",
                    "dataset_source": "TEDBench"
                }
            ],
            
            "attribute_modification": [
                {
                    "id": "A_001",
                    "image_path": "datasets/images/simple/attribute_modification/A_001.jpg",
                    "source_prompt": "A white horse",
                    "target_prompt": "A black horse",
                    "instruction": "Change it to a black horse",
                    "dataset_source": "TEDBench"
                },
                {
                    "id": "A_002",
                    "image_path": "datasets/images/simple/attribute_modification/A_002.jpg",
                    "source_prompt": "A fresh banana",
                    "target_prompt": "A rotten banana",
                    "instruction": "Make the banana rotten",
                    "dataset_source": "TEDBench"
                }
            ],
            
            "non_rigid_action": [
                {
                    "id": "N_001",
                    "image_path": "datasets/images/simple/non_rigid_action/N_001.jpg",
                    "source_prompt": "A photo of a blue and orange bird resting on a wooden branch",
                    "target_prompt": "A photo of a blue and orange bird spreading its wings on a wooden branch",
                    "instruction": "Make the bird spread its wings",
                    "dataset_source": "TEDBench"
                },
                {
                    "id": "N_002",
                    "image_path": "datasets/images/simple/non_rigid_action/N_002.jpg",
                    "source_prompt": "A photo of a closed brown wooden door on a dark grey wall",
                    "target_prompt": "A photo of an open brown wooden door on a dark grey wall",
                    "instruction": "Open the door",
                    "dataset_source": "TEDBench"
                }
            ],
            
            "style_transfer": [
                {
                    "id": "S_001",
                    "image_path": "datasets/images/simple/style_transfer/S_001.jpg",
                    "source_prompt": "A realistic photo of a cat",
                    "target_prompt": "A cartoon illustration of a cat",
                    "instruction": "Make it a cartoon",
                    "dataset_source": "IP2P"
                },
                {
                    "id": "S_002",
                    "image_path": "datasets/images/simple/style_transfer/S_002.jpg",
                    "source_prompt": "A beautiful landscape of snow-capped mountains and red hills at sunset",
                    "target_prompt": "A beautiful landscape of snow-capped mountains and red hills at sunrise",
                    "instruction": "change the sunset to a sunrise",
                    "dataset_source": "IP2P"
                }
            ],
            
            "reasoning": [
                {
                    "id": "R_001",
                    "image_path": "datasets/images/simple/reasoning/R_001.jpg",
                    "source_prompt": "A man wearing a casual t-shirt",
                    "target_prompt": "A man going to a formal funeral",
                    "instruction": "Make him look like he is going to a formal funeral",
                    "dataset_source": "Inter"
                }
            ] 
        }
    }

    # 写入json
    print("正在生成 Hybrid-EditBench ...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(seed_data, f, indent=4, ensure_ascii=False)

    print(f"生成成功，配置文件已保存至: {output_json_path}")

if __name__ == "__main__":
    build_dataset_template()