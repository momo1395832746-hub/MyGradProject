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

def register_new_cloud_model(provider, api_name, display_name):
    """
    核心控制网关：实现黑盒大模型的动态在线热插拔接驳
    """
    if not api_name or not display_name:
        return " 错误：请完整填写 API 标称与系统显示名称！", gr.update(), gr.update(), gr.update(), gr.update()
    
    if display_name in METHOD_REGISTRY:
        return f" 提示：模型名称 [{display_name}] 已存在，请勿重复注册。", gr.update(), gr.update(), gr.update(), gr.update()

    # 1. 动态注入路由注册表 (复用百炼云端评测管线)
    METHOD_REGISTRY[display_name] = "scripts/eval/qwen_edit_run.py"
    
    # 2. 动态实例化新模型并热插拔注入后端
    all_editors = get_editors()
    all_editors[display_name] = QwenEditEditor(model_name=api_name)
    
    # 3. 重新捕获最新的全局模型 choices 列表
    new_choices = list(METHOD_REGISTRY.keys())
    
    # 4. 重新构建大盘表格
    status_data = []
    for m in new_choices:
        m_type = "云端黑盒 API" if m not in ["InstructPix2Pix", "Pix2Pix-Zero", "SDEdit"] else "本地白盒 Sandbox"
        m_status = "🟢 已就绪 (Active)"
        status_data.append([m, m_type, m_status])
    new_status_df = pd.DataFrame(status_data, columns=["模型名称", "架构类型", "运行状态"])
    
    log_msg = f" 成功在线接入新模型！\n【供应商】: {provider}\n【API名称】: {api_name}\n【系统代号】: {display_name}"
    
    # 5. 跨街区组件联动升级 (关键手术：一口气刷新 Tab1, Tab2, Tab3 的所有模型容器)
    return (
        log_msg,
        new_status_df,
        gr.update(choices=new_choices, value=display_name), # 刷新 Tab 1 单图下拉框
        gr.update(choices=new_choices),                     # 刷新 Tab 2 评测下拉框
        gr.update(choices=new_choices)                      # 刷新 Tab 3 横评复选框
    )

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
        "num_inference_steps": int(steps) if steps else 50,
        "guidance_scale": float(txt_cfg) if txt_cfg else 7.5,
        "image_guidance_scale": float(img_cfg) if img_cfg else 1.5,
        "cross_attention_guidance_amount": float(cag_amount) if cag_amount else 0.0
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
    dataset_filename = os.path.basename(dataset_path)
    #  动态挂载专属工作区目录
    dynamic_cache_dir = os.path.join(PROJECT_ROOT, "datasets", "results", dataset_filename)
    os.makedirs(dynamic_cache_dir, exist_ok=True)

    
    # 转换内部标识符
    model_short_name = {"InstructPix2Pix": "ip2p", "Pix2Pix-Zero": "p2pz", "SDEdit": "sd", "Qwen-Edit": "qwen"}.get(method_name, 
        method_name.replace(" ", "_").replace("-", "_").lower())
    
    # 定义预期的 JSON 结果文件路径
    # 因为已经按文件夹隔离，文件名直接叫 ip2p.json 即可，干净清爽！
    json_file_name = f"{model_short_name}.json" 
    json_path = os.path.join(dynamic_cache_dir, json_file_name)

    # 【新增】定义延迟数据的读取路径
    latency_file = os.path.join(PROJECT_ROOT, "datasets", "results", "latency", f"{model_short_name}_latency.json")

    # 【新增】内部辅助函数：获取并计算平均耗时文本
    def get_avg_latency_str():
        if os.path.exists(latency_file):
            try:
                with open(latency_file, "r", encoding="utf-8") as lf:
                    lat_data = json.load(lf)
                latencies = list(lat_data.values())
                if latencies:
                    avg_l = sum(latencies) / len(latencies)
                    return f"{avg_l:.2f} 秒 / 图"
            except Exception:
                return "数据解析失败"
        return "暂无实测数据 (请确保已运行耗时测试)"
    
    # --- 阶段 0: 检查评测结果缓存 ---
    if os.path.exists(json_path):
        yield " 检测到该模型已存在评测结果，正在直接加载缓存数据...", None, None, None, None, None, None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            macro_df = pd.DataFrame(data["macro_metrics"])
            micro_df = pd.DataFrame(data["micro_metrics"])
            
            # 重新生成可视化图像（确保 UI 刷新）
            fig_dclip, fig_lpips = generate_single_model_bar_charts(micro_df, method_name)
            fig_radar_single = generate_vlm_perception_radar([method_name], "全部 (All)", dynamic_cache_dir)

            # 【核心修改】成功加载缓存时，顺便计算并返回平均延时
            avg_latency_str = get_avg_latency_str()
            
            yield " 成功加载历史评测数据！", macro_df, micro_df, fig_dclip, fig_lpips, fig_radar_single, avg_latency_str
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
        yield f"阶段 1/2: 地毯式搜索发现实存图像 {len(found_images)} 张，跳过推理环节。", None, None, None, None, None, "等待计算..."
    else:
        #  强制执行推理
        yield f" 阶段 1/2: 文件夹为空或只有子目录，准备启动终端生成脚本...", None, None, None, None, None, "运行推理中..."
        try:
            script_full_path = os.path.normpath(os.path.join(PROJECT_ROOT, script_rel_path))

            #  核心补丁：动态组装终端指令
            cmd = [sys.executable, script_full_path, "--dataset", dataset_path]
            # 如果发现调用的是云端 API 评测脚本，把内存里的真实 API 名字和新文件夹名穿透传给它！
            if "qwen_edit_run.py" in script_rel_path:
                editor_instance = get_editors().get(method_name)
                # 获取对象中真实的 API 模型名，若无则兜底
                real_api_name = getattr(editor_instance, 'model_name', 'qwen-image-edit-plus')
                # 追加命令行参数
                cmd.extend(["--api_name", real_api_name, "--save_dir_name", model_short_name])
        
            
            print(f"\n[终端指令] 正在执行: {sys.executable} {script_full_path} --dataset {dataset_path}", flush=True)
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False 
            )
            
            yield " 生成脚本执行完毕，正在转交 VLM 评测...", None, None, None, None, None, None
            
        except Exception as e:
            yield f" 脚本执行异常: {str(e)}", None, None, None, None, None, "运行出错"
            return

    # --- 阶段 2: 指标计算 ---
    yield f" 阶段 2/2: 正在进行指标计算与 VLM 评测...", None, None, None, None, None, None
    try:
        macro_df, micro_df = run_evaluation(
            model_name=model_short_name, 
            dataset_name=dataset_display_name, 
            dataset_path=dataset_path,
            model_display_name=method_name 
        )
        yield f" 正在启动 MLLM 语义审计 (调用 Qwen3.5-Plus)... 请切回终端查看进度条！", macro_df, micro_df, None, None, None, "VLM审计中..."
        
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
            yield f" VLM 裁判罢工了 (API报错或断网)，请检查终端日志。目前仅展示客观指标。", macro_df, micro_df, None, None, None, "部分完成"
            # 注意：这里不 return，即便 VLM 失败，依然可以画出客观指标的柱状图
        
        # 3. 重新加载融合后的数据进行前端绘图
        yield f" 所有数据计算完毕，正在绘制分析图表...", macro_df, micro_df, None, None, None, "图表绘制中..."
        
        fig_dclip, fig_lpips = generate_single_model_bar_charts(micro_df, method_name)
        
        fig_radar_single = generate_vlm_perception_radar([method_name], "全部 (All)", dynamic_cache_dir)

        # 【核心修改】全套流程跑完，计算最终实测平均延时
        avg_latency_str = get_avg_latency_str()
        yield f" 评测成功！新结果已存入硬盘。", macro_df, micro_df, fig_dclip, fig_lpips, fig_radar_single, avg_latency_str
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f" 评测过程中发生错误: {str(e)}", None, None, None, None, None, "运行出错"

# --- 模块三：多方法横评 ---
def refresh_and_compare(selected_methods, complexity_level, task_filter, dataset_display_name):
    if not selected_methods:
        return "请至少选择一个模型", None, None
    
    try:

        # 👉 动态定位横评的目标文件夹
        dataset_path = DATASET_REGISTRY.get(dataset_display_name)
        dataset_filename = os.path.basename(dataset_path)
        target_dir = os.path.join(PROJECT_ROOT, "datasets", "results", dataset_filename)

        # 直接把结果目录传给引擎
        fig_vlm = generate_vlm_perception_radar(selected_methods, complexity_level, target_dir)
        fig_pareto = generate_pareto_scatter_plot(selected_methods, complexity_level, task_filter, target_dir)
        
        return "分析图表已成功生成！", fig_vlm, fig_pareto
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"绘图失败: {str(e)}", None, None

# ================== UI 搭建 ==================
with gr.Blocks(title="指令驱动图像编辑评测平台", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("##  基于扩散模型的指令驱动图像编辑方法研究与比较平台")
    gr.Markdown("欢迎使用本平台。")
    
    with gr.Tabs():
        # 👇=== 全新加入的 Tab 0：系统初始状态与云端动态接驳面板 ===👇
        with gr.TabItem(" 平台概览与在线云端模型接入"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("###  当前系统就绪模型大盘")
                    # 动态生成初始表格数据
                    init_data = [
                        ["InstructPix2Pix", "本地模型", "🟢 已就绪 (Active)"],
                        ["Pix2Pix-Zero", "本地模型", "🟢 已就绪 (Active)"],
                        ["SDEdit", "本地模型", "🟢 已就绪 (Active)"],
                        ["Qwen-Edit", "云端模型", "🟢 已就绪 (Active)"]
                    ]
                    init_df = pd.DataFrame(init_data, columns=["模型名称", "架构类型", "运行状态"])
                    model_status_market = gr.Dataframe(value=init_df, interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("###  动态接入新云端模型 ")
                    config_provider = gr.Dropdown(choices=["阿里百炼 (DashScope)"], value="阿里百炼 (DashScope)", label="1. 选择云端能力供应商")
                    config_api_name = gr.Textbox(placeholder="如: qwen-image-edit-v2", label="2. 输入目标模型 API 标称")
                    config_display_name = gr.Textbox(placeholder="如: Qwen-Edit-V2", label="3. 设定系统内唯一显示名称")
                    
                    config_action_btn = gr.Button(" 激活并接入平台生态", variant="primary")
                    config_log = gr.Textbox(label="网络接驳控制台日志", lines=4, interactive=False)
                    
        #模块一: 单图交互编辑
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

                # 【核心修改】在左侧面板最下方增设“实际工程计算效率”展示框
                    m2_latency = gr.Textbox(
                        label=" 实际计算效率 (端到端推理延迟)", 
                        placeholder="等待评测激活...", 
                        interactive=False
                    )
                    
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


            
        # 模块 3: 多方法横向对比
        with gr.TabItem("3. 多方法综合对比分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  横评控制台")
                    
                    m3_dataset = gr.Dropdown(choices=list(DATASET_REGISTRY.keys()), value="mini 测试集 (推荐调试用)", label="选择要对比的数据集结果")
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

    # =========================================================================
    # 🎯 核心事件路由总机 (Event Router)
    # 将所有的 click 和 change 事件统一放置在 Blocks 作用域的最底部！
    # 此时所有的 UI 组件都已在内存中实例化完毕，安全可靠，零报错。
    # =========================================================================
    
    # 1. 🌐 Tab 0: 云端模型动态接驳事件
    config_action_btn.click(
        fn=register_new_cloud_model,
        inputs=[config_provider, config_api_name, config_display_name],
        outputs=[config_log, model_status_market, model_dropdown, m2_method, m3_methods] 
    )

    # 2. 🎨 Tab 1: 单图交互参数联动与执行事件
    model_dropdown.change(fn=update_ui, inputs=[model_dropdown], outputs=[source_prompt_input, steps_slider, txt_cfg_slider, img_cfg_slider, cag_amount_slider])
    m1_run_btn.click(fn=process_edit, inputs=[input_image, prompt_input, source_prompt_input, model_dropdown, steps_slider, txt_cfg_slider, img_cfg_slider, cag_amount_slider], outputs=output_image)

    # 3. 📈 Tab 2: 单方法基准测试执行事件
    m2_bench_btn.click(run_benchmark_pipeline, [m2_method, m2_dataset], [m2_status, macro_table, micro_table, bar_dclip, bar_lpips, radar_vlm_single, m2_latency])

    # 4. ⚖️ Tab 3: 多方法交叉横评图表生成事件
    m3_compare_btn.click(refresh_and_compare, [m3_methods, m3_complexity, m3_task_filter, m3_dataset], [m3_status, radar_vlm, scatter_pareto])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)