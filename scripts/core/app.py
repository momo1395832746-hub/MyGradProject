import gradio as gr
import subprocess
import os
import sys
import pandas as pd
import json
import traceback

# 自动获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets", "results")
sys.path.append(PROJECT_ROOT)

try:
    from scripts.core.editors import InstructPix2PixEditor, Pix2PixZeroEditor, SDEditEditor, QwenEditEditor
    from scripts.eval.eval_metrics import run_evaluation
    from scripts.eval.visualizer import (
        generate_single_model_bar_charts, 
        generate_vlm_perception_radar,
        generate_pareto_scatter_plot
    )
except ImportError as e:
    print(f" 模块导入警告: {e}")

# 注册表配置
METHOD_REGISTRY = {
    "InstructPix2Pix": "scripts/eval/ip2p_run.py",
    "Pix2Pix-Zero": "scripts/eval/p2pz_run.py",
    "SDEdit": "scripts/eval/sd_run.py",
    "Qwen-Edit": "scripts/eval/qwen_edit_run.py"
}

DATASET_REGISTRY = {
    "mini 测试集 (推荐调试用)": os.path.join(PROJECT_ROOT, "datasets", "Hybrid-EditBench-mini.json"),
    "完整基准集 (耗时长)": os.path.join(PROJECT_ROOT, "datasets", "Hybrid-EditBench.json")
}

def load_cached_results():
    """从硬盘读取所有已保存的评测 JSON"""
    all_results = {}
    if not os.path.exists(CACHE_DIR):
        return all_results
    
    for file_name in os.listdir(CACHE_DIR):
        if file_name.endswith(".json"):
            try:
                with open(os.path.join(CACHE_DIR, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 我们主要需要微观数据表来绘图
                    method_name = data.get("model_name_display", data["model_name"])
                    all_results[method_name] = pd.DataFrame(data["micro_metrics"])
            except Exception as e:
                print(f"读取缓存失败 {file_name}: {e}")
    return all_results
    
editors = {}
def get_editors():
    global editors
    if not editors:
        editors = {"SDEdit": SDEditEditor(), "InstructPix2Pix": InstructPix2PixEditor(), "Pix2Pix-Zero": Pix2PixZeroEditor(), "Qwen-Edit": QwenEditEditor()}
    return editors

# ==========================================
# 交互编辑回调函数 (用于 Tab 1)
# ==========================================
def process_edit(input_image, prompt, source_prompt, model_choice, steps, txt_cfg, img_cfg, cag_amount):
    """
    处理单图交互式编辑的统一网关函数
    """
    # 1. 基础输入校验 (防御性编程，防止页面崩溃)
    if input_image is None:
        raise gr.Error(" 请先上传一张需要编辑的原始图片！")
    if not prompt or prompt.strip() == "":
        raise gr.Error(" 请输入明确的编辑指令 (Prompt)！")

    # 2. 获取对应的编辑器实例
    # 这里的 editors 字典应该在你 app.py 的顶部已经定义好了
    # 例如: editors = {"InstructPix2Pix": InstructPix2PixEditor(), "Qwen-Edit(云端)": QwenEditEditor()}
    editor = get_editors().get(model_choice)
    if not editor:
        raise gr.Error(f" 抱歉，模型 [{model_choice}] 尚未注册或未成功加载。")

    # 3. 组装高级参数
    # 不同的模型需要的参数不同。云端大模型可能只需要 prompt，
    # 而本地 Diffusion 模型需要 steps, cfg 等。通过 kwargs 统一打包。
    kwargs = {
        "source_prompt": source_prompt,
        "steps": int(steps) if steps else 50,
        "txt_cfg": float(txt_cfg) if txt_cfg else 7.5,
        "img_cfg": float(img_cfg) if img_cfg else 1.5,
        "cag_amount": float(cag_amount) if cag_amount else 0.0
    }

    print(f" 开始单图调试: 使用模型 [{model_choice}]...")
    print(f" 指令: {prompt}")

    # 4. 执行编辑与异常捕获
    try:
        # 调用对应 Editor 类的 edit 方法
        result_image = editor.edit_image(image=input_image, prompt=prompt, **kwargs)
        
        if result_image is None:
             raise gr.Error("模型返回了空图像，请检查终端报错日志。")
             
        print(" 单图生成成功！")
        return result_image
        
    except Exception as e:
        # 在前端弹出红色错误提示框，同时在终端打印详细堆栈方便你 debug
        import traceback
        traceback.print_exc() 
        raise gr.Error(f"编辑过程中发生错误: {str(e)}")

def update_ui(model_choice):
    if model_choice == "InstructPix2Pix": return [gr.update(visible=False), gr.update(visible=True, value=20), gr.update(visible=True, value=7.5), gr.update(visible=True, value=1.5), gr.update(visible=False)]
    elif model_choice == "Pix2Pix-Zero": return [gr.update(visible=True), gr.update(visible=True, value=50), gr.update(visible=True, value=5.0), gr.update(visible=False), gr.update(visible=True, value=0.1)]
    elif model_choice == "SDEdit": return [gr.update(visible=False), gr.update(visible=True, value=50), gr.update(visible=True, value=7.5), gr.update(visible=False), gr.update(visible=False)]
    elif model_choice == "Qwen-Edit": 
        #黑盒云端 API：隐藏所有底层扩散模型的超参数滑块
        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
    return [gr.update() for _ in range(5)]

# ================== Benchmark 调度 ==================
def run_benchmark_pipeline(method_name, dataset_display_name):
    script_rel_path = METHOD_REGISTRY.get(method_name)
    dataset_path = DATASET_REGISTRY.get(dataset_display_name)
    dataset_short_name = "mini" if "Mini" in dataset_display_name else "full"
    
    # 转换内部标识符
    model_short_name = {"InstructPix2Pix": "ip2p", "Pix2Pix-Zero": "p2pz", "SDEdit": "sd", "Qwen-Edit": "qwen"}.get(method_name)
    
    # 定义预期的 JSON 结果文件路径
    # 这里的命名逻辑要和 run_evaluation 内部存盘逻辑保持一致
    json_file_name = f"{model_short_name}_{dataset_short_name}.json"
    json_path = os.path.join(CACHE_DIR, json_file_name)
    
    # --- 阶段 0: 检查评测结果缓存 ---
    if os.path.exists(json_path):
        yield " 检测到该模型已存在评测结果，正在直接加载缓存数据...", None, None, None, None, None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            macro_df = pd.DataFrame(data["macro_metrics"])
            micro_df = pd.DataFrame(data["micro_metrics"])
            
            # 重新生成可视化图像（确保 UI 刷新）
            fig_dclip, fig_lpips = generate_single_model_bar_charts(micro_df, method_name)
            fig_radar_single = generate_vlm_perception_radar([method_name], "全部 (All)", CACHE_DIR)
            
            yield " 成功加载历史评测数据！", macro_df, micro_df, fig_dclip, fig_lpips, fig_radar_single
            return # 任务结束
        except Exception as e:
            print(f"读取缓存 JSON 失败: {e}，将重新执行评测...")

    # --- 阶段 1: 检查推理图像缓存 ---
    # 假设你的推理脚本生成的图片放在这个目录下
    search_paths = [os.path.join(PROJECT_ROOT, "datasets", "generated", model_short_name),]  
    found_images = []
    actual_dir = ""
    
    for path in search_paths:
        if os.path.exists(path):
            valid_imgs = []
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        valid_imgs.append(os.path.join(root, f))
            
            if len(valid_imgs) > 0:
                found_images = valid_imgs
                actual_dir = path
                break

    if len(found_images) > 0:
        yield f"阶段 1/2: 地毯式搜索发现实存图像 {len(found_images)} 张，跳过推理环节。", None, None, None, None, None
    else:
        #  强制执行推理
        yield f" 阶段 1/2: 文件夹为空或只有子目录，准备启动终端生成脚本...", None, None, None, None, None
        try:
            script_full_path = os.path.normpath(os.path.join(PROJECT_ROOT, script_rel_path))
            print(f"\n[终端指令] 正在执行: {sys.executable} {script_full_path} --dataset {dataset_path}", flush=True)
            
            result = subprocess.run(
                [sys.executable, script_full_path, "--dataset", dataset_path],
                check=True,
                capture_output=False 
            )
            
            yield " 生成脚本执行完毕，正在转交 VLM 评测...", None, None, None, None, None
            
        except Exception as e:
            yield f" 脚本执行异常: {str(e)}", None, None, None, None, None
            return

    # --- 阶段 2: 指标计算 ---
    yield f" 阶段 2/2: 正在进行指标计算与 VLM 评测...", None, None, None, None, None
    try:
        macro_df, micro_df = run_evaluation(
            model_name=model_short_name, 
            dataset_name=dataset_short_name, 
            dataset_path=dataset_path,
            model_display_name=method_name 
        )
        yield f" 正在启动 MLLM 语义审计 (调用 Qwen3.5-Plus)... 请切回终端查看进度条！", macro_df, micro_df, None, None, None
        
        try:
            # 获取脚本绝对路径
            vlm_script_path = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts", "eval", "vlm_judge.py"))
            dataset_filename = os.path.basename(dataset_path) # 取文件名，如 Hybrid-EditBench-mini.json
            
            # 使用 sys.executable 作为独立子进程启动，不锁死网页内存
            print(f"\n[终端指令] 唤醒大模型裁判: {sys.executable} {vlm_script_path} --model_name {model_short_name} --dataset {dataset_filename}", flush=True)
            
            # 必须把 api-key 的环境变量继承给子进程
            env = os.environ.copy()
            # 如果你在终端 export 过，这里可以不加；如果没 export，建议在这里强制写入
            # env["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxx" 
            
            subprocess.run(
                [sys.executable, vlm_script_path, "--model_name", model_short_name, "--dataset", dataset_filename],
                check=True,
                capture_output=False, # 让 tqdm 进度条在您的黑框终端里疯狂跳动
                env=env
            )
            print(" VLM 裁判打分完毕！")
        except subprocess.CalledProcessError as e:
            yield f" VLM 裁判罢工了 (API报错或断网)，请检查终端日志。目前仅展示客观指标。", macro_df, micro_df, None, None, None
            # 注意：这里不 return，即便 VLM 失败，依然可以画出客观指标的柱状图
        
        # 3. 重新加载融合后的数据进行前端绘图
        yield f" 所有数据计算完毕，正在绘制分析图表...", macro_df, micro_df, None, None, None
        
        fig_dclip, fig_lpips = generate_single_model_bar_charts(micro_df, method_name)
        
        fig_radar_single = generate_vlm_perception_radar([method_name], "全部 (All)", CACHE_DIR)
        
        
        yield f" 评测成功！新结果已存入硬盘。", macro_df, micro_df, fig_dclip, fig_lpips, fig_radar_single
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f" 评测过程中发生错误: {str(e)}", None, None, None, None, None

# --- 模块三：多方法横评 ---
def refresh_and_compare(selected_methods, complexity_level, task_filter):
    if not selected_methods:
        return "请至少选择一个模型", None, None
    
    try:
        # PROJECT_ROOT 和 CACHE_DIR 在你 app.py 顶部已经定义了
        # 直接把结果目录传给引擎
        fig_vlm = generate_vlm_perception_radar(selected_methods, complexity_level, CACHE_DIR)
        fig_pareto = generate_pareto_scatter_plot(selected_methods, complexity_level, task_filter, CACHE_DIR)
        
        return "分析图表已成功生成！", fig_vlm, fig_pareto
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"绘图失败: {str(e)}", None, None

# ================== UI 搭建 ==================
with gr.Blocks(title="指令驱动图像编辑评测平台", theme=gr.themes.Soft()) as demo:
    gr.Markdown("##  基于扩散模型的指令驱动图像编辑方法研究与比较平台")
    
    #模块一: 单图交互编辑
    with gr.Tabs():
        # Tab 1
        with gr.TabItem("1. 单图交互编辑 (调试)"):
            gr.Markdown("*(这里是单图编辑界面)*")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="上传原图", type="pil")
                    prompt_input = gr.Textbox(label="编辑指令")
                    source_prompt_input = gr.Textbox(label="源提示词", visible=False)
                    model_dropdown = gr.Dropdown(choices=list(METHOD_REGISTRY.keys()), value="InstructPix2Pix", label="选择模型")
                    with gr.Accordion("高级参数", open=True):
                        steps_slider = gr.Slider(10, 100, 20, step=1, label="Steps")
                        txt_cfg_slider = gr.Slider(1.0, 15.0, 7.5, step=0.5, label="Text CFG")
                        img_cfg_slider = gr.Slider(1.0, 3.0, 1.5, step=0.1, label="Image CFG")
                        cag_amount_slider = gr.Slider(0.0, 0.5, 0.15, step=0.01, label="CAG", visible=False)
                    m1_run_btn = gr.Button(" 开始单图编辑", variant="primary")
                with gr.Column(scale=1):
                    output_image = gr.Image(label="预览")
            model_dropdown.change(fn=update_ui, inputs=[model_dropdown], outputs=[source_prompt_input, steps_slider, txt_cfg_slider, img_cfg_slider, cag_amount_slider])
            m1_run_btn.click(fn=process_edit, inputs=[input_image, prompt_input, source_prompt_input, model_dropdown, steps_slider, txt_cfg_slider, img_cfg_slider, cag_amount_slider], outputs=output_image)

       # 模块二: 单方法 Benchmark 测试与可视化分析
        with gr.TabItem("2. 单方法 Benchmark 评测"):
            # ========== 第一层区块：左右分栏 (控制面板 vs 数据表) ==========
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  评测调度")
                    m2_method = gr.Dropdown(choices=list(METHOD_REGISTRY.keys()), value="InstructPix2Pix", label="1. 调度评测模型")
                    m2_dataset = gr.Dropdown(choices=list(DATASET_REGISTRY.keys()), value="mini 测试集 (推荐调试用)", label="2. 挂载测试数据集")
                    m2_bench_btn = gr.Button(" 启动该模型的专属评测", variant="primary")
                    m2_status = gr.Textbox(label="系统日志", lines=5, interactive=False)
                    
                with gr.Column(scale=2):
                    gr.Markdown("###  量化指标报表")
                    macro_table = gr.Dataframe(label="宏观复杂度表现")
                    micro_table = gr.Dataframe(label="微观任务细分表现")
                    
            gr.Markdown("---")
            gr.Markdown("*柱状图直观对比了该模型在不同难度（Simple vs Complex）和不同编辑任务下的具体指标差异。*")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("###  主观感知分析 (VLM Radar)")

                    gr.HTML("<div style='height: 70px;'></div>")

                    radar_vlm_single = gr.Plot(show_label=False)
                with gr.Column(scale=3):
                    gr.Markdown("###  客观指标分布 (Bar Charts)")
                    bar_dclip = gr.Plot(show_label=False)
                    bar_lpips = gr.Plot(show_label=False)

            # ========== 事件绑定 (保持不变) ==========
            m2_bench_btn.click(
                run_benchmark_pipeline,
                [m2_method, m2_dataset],
                [m2_status, macro_table, micro_table, bar_dclip, bar_lpips, radar_vlm_single]
            )
            
        # 模块 3: 多方法横向对比
        with gr.TabItem("3. 多方法综合对比分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  横评控制台")
                    m3_complexity = gr.Radio(choices=["全部 (All)", "简单 (Simple)", "复杂 (Complex)"], value="全部 (All)", label="评估基准场景")

                    task_choices = [
                        "Object Replacement", 
                        "Attribute Modification", 
                        "Style Transfer", 
                        "Non Rigid Action", 
                        "Reasoning"
                    ]
                    m3_task_filter = gr.CheckboxGroup(
                        choices=task_choices, 
                        value=task_choices, # 默认把这 5 个都勾上，代表全选
                        label="筛选特定编辑任务 (仅对帕累托图生效，可多选)"
                    )
                    
                    m3_methods = gr.CheckboxGroup(choices=list(METHOD_REGISTRY.keys()), label="选择对比模型")
                    m3_compare_btn = gr.Button(" 融合生成分析图表", variant="primary")
                    m3_status = gr.Textbox(label="系统日志", interactive=False)
                    
                with gr.Column(scale=3):
                    gr.Markdown("###  编辑保真度权衡 (Pareto Frontier)")
                    scatter_pareto = gr.Plot(show_label=False)
                    gr.Markdown("---")
                    gr.Markdown("###  VLM 主观感知雷达图呈列")
                    radar_vlm = gr.Plot(show_label=False) 

            m3_compare_btn.click(
                refresh_and_compare,
                [m3_methods, m3_complexity, m3_task_filter],
                [m3_status, radar_vlm, scatter_pareto] # 需要在 refresh_and_compare 里按这个顺序返回
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)