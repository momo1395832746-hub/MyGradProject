import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
import os
import json
import math
import glob
from matplotlib.font_manager import FontProperties

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
font_path = os.path.join(project_root, "SimHei.ttf")
my_font = FontProperties(fname=font_path)

def plot_multi_model_radar(models_data, categories, title, metric_name, y_max=1.0, save_path=None):
    """
    绘制多模型对比雷达图 (升级版：支持动态量纲 0-1 或 1-5)
    :param models_data: 字典格式，如 {'InstructPix2Pix': [4.0, 3.5, 4.2], 'P2P-Zero': [...]}
    :param categories: 任务类别列表，即雷达图的各个顶点
    :param title: 图表标题
    :param metric_name: 指标名称
    :param y_max: Y轴最大值 (VLM打分传 5.0，D-CLIP/LPIPS传 1.0)
    """
    N = len(categories)
    if N < 3:
        print("[警告] 雷达图顶点少于3个，无法成图！")
        return None
        
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(pi / 2) 
    ax.set_theta_direction(-1)  
    plt.xticks(angles[:-1], categories, color='black', size=12, fontweight='bold')
    
    ax.set_rlabel_position(0)
    plt.ylim(0, y_max)
    
    # 动态设置刻度
    if y_max == 5.0:
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
    else:
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    
    # 学术配色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (model_name, values) in enumerate(models_data.items()):
        values = list(values)
        values += values[:1]
        
        ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=2.5, linestyle='solid', 
                marker=markers[idx % len(markers)], markersize=8, label=model_name)
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.15)
    
    plt.title(title, size=16, color='black', y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

# ================= 模拟数据测试 =================
if __name__ == "__main__":
    # 这是我们在跑完所有模型后，从各个 micro_df 中提取整合出来的数据
    # 这里的类别必须与你 Hybrid-EditBench 中的 Task 严格对应
    task_categories = ['Attribute', 'Structure', 'Object Replace', 'Style', 'Reasoning']
    title="D-CLIP Score - Object-Centric"
    
    # 模拟“简单图像 (Object-centric)”下的 D-CLIP 得分
    simple_image_dclip_data = {
        'InstructPix2Pix': [0.25, 0.15, 0.28, 0.35, 0.10],
        'Pix2Pix-Zero': [0.22, 0.18, 0.20, 0.30, 0.08],
        'SD Baseline': [0.10, 0.05, 0.12, 0.15, 0.02]
    }
    
    # 模拟“复杂图像 (Compositional)”下的 D-CLIP 得分 (普遍分数会下降)
    complex_image_dclip_data = {
        'InstructPix2Pix': [0.18, 0.10, 0.20, 0.25, 0.05],
        'Pix2Pix-Zero': [0.15, 0.12, 0.15, 0.22, 0.04],
        'SD Baseline': [0.05, 0.02, 0.08, 0.10, -0.05] # 复杂场景可能会出现负分
    }
    
    # 生成 简单图像 雷达图
    fig1 = plot_multi_model_radar(
        models_data=simple_image_dclip_data,
        categories=task_categories,
        title="D-CLIP 语义一致性对比 - 简单图像 (Object-centric)",
        metric_name="D-CLIP",
        save_path="results/figures/radar_simple_dclip.png"
    )
    
    # 生成 复杂图像 雷达图
    fig2 = plot_multi_model_radar(
        models_data=complex_image_dclip_data,
        categories=task_categories,
        title="D-CLIP 语义一致性对比 - 复杂图像 (Compositional)",
        metric_name="D-CLIP",
        save_path="results/figures/radar_complex_dclip.png"
    )

def generate_vlm_perception_radar(selected_methods, complexity_level, results_dir):
    """
    模块三专用：生成动态网格布局的 VLM 感知雷达图
    """
    name_mapping = {"InstructPix2Pix": "ip2p", "Pix2Pix-Zero": "p2pz", "SDEdit": "sd", "Qwen-Edit": "qwen"}
    valid_methods = [m for m in selected_methods if name_mapping.get(m)]
    if not valid_methods: return None
        
    num_models = len(valid_methods)
    
    # 核心排版升级：动态计算行列数 (最多两列，自动换行)
    cols = 2 if num_models >= 2 else 1
    rows = math.ceil(num_models / cols)
    
    # 动态分配画布大小，每个子图占据 6x6 的充裕空间
    fig = plt.figure(figsize=(7 * cols, 6.5 * rows))
    
    plot_idx = 1
    for method in valid_methods:
        short_name = name_mapping.get(method)
        file_path = os.path.join(results_dir, "vl_results", f"vlm_scores_{short_name}.json")
        
        if not os.path.exists(file_path):
            plot_idx += 1
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            task_data = {}
            for img_id, metrics in data.items():
                comp = metrics.get('complexity', '').lower()
                cat = metrics.get('category', 'unknown').replace('_', '\n').title() # 换行防拥挤
                
                if "简单" in complexity_level and comp != "simple": continue
                if "复杂" in complexity_level and comp != "complex": continue
                
                if cat not in task_data: task_data[cat] = {'EQ': [], 'OPP': [], 'ICP': []}
                task_data[cat]['EQ'].append(metrics.get('EQ', 0))
                task_data[cat]['OPP'].append(metrics.get('OPP', 0))
                task_data[cat]['ICP'].append(metrics.get('ICP', 0))
                
            if not task_data:
                plot_idx += 1
                continue
                
            categories = sorted(list(task_data.keys()))
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            # 使用计算好的 rows 和 cols
            ax = fig.add_subplot(rows, cols, plot_idx, polar=True)
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            
            plt.xticks(angles[:-1], categories, color='black', size=11, fontweight='bold')
            ax.set_rlabel_position(0)
            plt.ylim(0, 5.0)
            plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="#999999", size=9)
            
            metrics_to_plot = ['EQ', 'OPP', 'ICP']
            colors = ['#d62728', '#1f77b4', '#2ca02c']
            markers = ['o', 's', '^']
            
            for m_idx, metric in enumerate(metrics_to_plot):
                values = [sum(task_data[c][metric])/len(task_data[c][metric]) if task_data[c][metric] else 0 for c in categories]
                values += values[:1]
                ax.plot(angles, values, color=colors[m_idx], linewidth=2, linestyle='solid', marker=markers[m_idx], markersize=5, label=f"{metric}")
                ax.fill(angles, values, color=colors[m_idx], alpha=0.15)
                
            plt.title(f"{method}", size=15, color='black', y=1.12, fontweight='bold')
            if plot_idx == 1: # 只在第一个图显示图例，让画面更干净
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)
                
        except Exception as e: print(f"[解析错误] {method}: {e}")
        plot_idx += 1

    plt.tight_layout(pad=3.0) # 增加子图间距

    # ================= 新增：雷达图全自动保存图像逻辑 =================
    try:
        import time 
        
        # 1. 智能动态识别模型标签 (防 NameError 核心手术)
        model_label = "Model"
        for var_name in ['method', 'short_name', 'method_name', 'model_name', 'title']:
            if var_name in locals() and locals()[var_name]:
                # 过滤掉可能包含特殊字符的长标题，只取前15个字符
                model_label = str(locals()[var_name]).split('\n')[0][:15]
                break
                
        # 2. 智能识别保存目录，若无则默认输出至当前目录 '.'
        out_dir = results_dir if 'results_dir' in locals() else '.'
        
        # 生成安全的文件名（包含模型标签与时间戳）
        safe_label = model_label.replace(" ", "_").replace("/", "_").replace("-", "_")
        timestamp = int(time.time())
        base_filename = f"Radar_{safe_label}_{timestamp}"
        
        save_png_path = os.path.join(out_dir, f"{base_filename}.png")
        save_pdf_path = os.path.join(out_dir, f"{base_filename}.pdf")
        
        # 3. 智能获取当前画布对象
        current_fig = fig if 'fig' in locals() else plt.gcf()
        
        # 4. 统一导出高清 PNG (dpi=400) 和 矢量 PDF 格式
        current_fig.savefig(save_png_path, dpi=400, bbox_inches='tight')
        current_fig.savefig(save_pdf_path, bbox_inches='tight')
        print(f"✅ 雷达图已成功保存至: \n   - {save_png_path}\n   - {save_pdf_path}")
    except Exception as e:
        print(f"⚠️ 雷达图保存失败: {e}")
    # ==================================================================
    
    return fig

def plot_metric_bar_chart(df, metric_col, title, ylabel,ylim=None):
    """
    绘制分组柱状图：X轴为Task，两组柱子分别为 Simple 和 Complex
    """
    tasks = df['Task'].unique().tolist()
    simple_scores = []
    complex_scores = []
    
    # 宽容匹配 Level
    for task in tasks:
        task_df = df[df['Task'] == task]
        s_val = task_df[task_df['Level'].str.upper().isin(['SIMPLE', 'OBJECT-CENTRIC'])][metric_col]
        c_val = task_df[task_df['Level'].str.upper().isin(['COMPLEX', 'COMPOSITIONAL'])][metric_col]
        
        simple_scores.append(s_val.values[0] if not s_val.empty else 0)
        complex_scores.append(c_val.values[0] if not c_val.empty else 0)
        
    x = np.arange(len(tasks))
    width = 0.35  # 柱子宽度
    
    #  核心手术 1：极限压扁拉宽！从 (10, 4.5) 改为 (12, 3.2)
    # 高度变成 3.2，两张图叠起来是 6.4，刚好和左侧雷达图的 6.5 高度完美对齐对齐！
    fig, ax = plt.subplots(figsize=(12, 5.0))
    
    # 采用学术界常用的莫兰迪配色体系
    ax.bar(x - width/2, simple_scores, width, label='Simple (Object-centric)', color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, complex_scores, width, label='Complex (Compositional)', color='#DD8452', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    #  核心手术 2：减小标题的 pad (边距)，节省顶部空间
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    #  核心手术 3：X轴标签倾斜角度减小到 25度，防止文字往下掉撑破画布
    ax.set_xticklabels(tasks, rotation=35, ha='right', fontsize=10)
    
    # 添加网格线，辅助阅读
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # 让网格线在柱子下方
    
    # 图例缩小一点，放在最佳位置
    ax.legend(fontsize=10, loc='upper right')

    if ylim is not None:
        ax.set_ylim(ylim)
    # 自动紧凑布局
    plt.tight_layout()
    
    return fig

def generate_single_model_bar_charts(micro_df, model_name):
    """
    接收 DataFrame，分别生成 D-CLIP 和 LPIPS 的两张独立柱状图
    """
    if micro_df.empty or 'Task' not in micro_df.columns:
        return None, None
        
    # 图表 1: D-CLIP (语义一致性) - 越高越好
    fig_dclip = plot_metric_bar_chart(
        df=micro_df, 
        metric_col='D-CLIP (↑)', 
        title=f'{model_name} - D-CLIP Semantic Consistency (Higher is Better)', 
        ylabel='D-CLIP Score',
        ylim=(0, 0.40)
    )
    
    # 图表 2: LPIPS (结构保真度) - 越低越好
    fig_lpips = plot_metric_bar_chart(
        df=micro_df, 
        metric_col='LPIPS (↓)', 
        title=f'{model_name} - LPIPS Structural Distance (Lower is Better)', 
        ylabel='LPIPS Distance',
        ylim=(0, 0.65)
    )
    
    return fig_dclip, fig_lpips

def generate_pareto_scatter_plot(selected_methods, complexity_level, task_filter, results_dir):
    """
    模块三专用：绘制 LPIPS (X轴) vs D-CLIP (Y轴) 散点图
    规避 LPIPS 的“欺骗数据”陷阱
    """
    name_mapping = {"InstructPix2Pix": "ip2p", "Pix2Pix-Zero": "p2pz", "SDEdit": "sd", "Qwen-Edit": "qwen"}
    valid_methods = [m for m in selected_methods if name_mapping.get(m)]
    if not valid_methods: return None

    # 如果用户把任务全取消勾选了，拦截并提示
    if not task_filter or len(task_filter) == 0:
        raise gr.Error("请至少保留一个任务类别！")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    has_data = False

    #用于收集全局数据，动态计算十字准线
    global_all_x = []
    global_all_y = []
    
    for idx, method in enumerate(valid_methods):
        short_name = name_mapping.get(method)
       # 修复 1：动态模糊查找 json 文件，不再写死 _full
        search_pattern = os.path.join(results_dir, f"{short_name}_*.json")
        # 排除掉 vlm_scores 文件
        possible_files = [f for f in glob.glob(search_pattern) if "vlm_scores" not in f]
        
        if not possible_files:
            print(f"[数据缺失] 找不到 {method} 的客观指标 JSON 文件")
            continue
            
        file_path = possible_files[0] # 取找到的第一个文件
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            x_lpips, y_dclip = [], []
            
            for item in data.get("micro_metrics", []):
                comp = item.get("Level", "").upper()
                task_name = item.get("Task", "").replace('_', ' ').title()
                
                # 难度过滤
                if "简单" in complexity_level and comp != "SIMPLE": continue
                if "复杂" in complexity_level and comp != "COMPLEX": continue
                
                if task_name not in task_filter: 
                    continue
                
                if "LPIPS (↓)" in item and "D-CLIP (↑)" in item:
                    x_lpips.append(item["LPIPS (↓)"])
                    y_dclip.append(item["D-CLIP (↑)"])
            
            if x_lpips and y_dclip:
                has_data = True
                
                global_all_x.extend(x_lpips)
                global_all_y.extend(y_dclip)
                
                ax.scatter(x_lpips, y_dclip, c=colors[idx % len(colors)], marker=markers[idx % len(markers)], 
                           s=120, alpha=0.8, edgecolors='white', label=method)
                mean_x, mean_y = sum(x_lpips)/len(x_lpips), sum(y_dclip)/len(y_dclip)
                ax.scatter(mean_x, mean_y, c=colors[idx % len(colors)], marker='P', s=300, edgecolors='black')
                
        except Exception as e: print(f"[解析错误] Pareto {method}: {e}")

    if not has_data: return None

    # 动态标题
    # 1. 定义您的全量任务库（用于比对）
    all_tasks = [
        "Object Replacement", 
        "Attribute Modification", 
        "Style Transfer", 
        "Non Rigid Action", 
        "Reasoning"
    ]
    
    # 2. 智能副标题生成逻辑
    if len(task_filter) == len(all_tasks):
        subtitle = "Tasks: All Included"
    else:
        # 战术：找出“被排除”的任务。通常排除的少，写出来更短更清晰
        excluded_tasks = [t for t in all_tasks if t not in task_filter]
        
        if len(excluded_tasks) <= 2:
            subtitle = f"Tasks Excluded: {', '.join(excluded_tasks)}"
        else:
            # 如果排除的太多，就干脆只显示数量，保持版面整洁
            subtitle = f"Filtered Tasks: {len(task_filter)} / {len(all_tasks)} Included"

    # 3. 部署主标题 (加大 padding 把上方空间腾出来给副标题)
    ax.set_title(f"Pareto Trade-off ({complexity_level})", fontproperties=my_font,fontsize=16, fontweight='bold', pad=25)
    
    # 4. 部署副标题 (利用相对坐标悬浮在主标题下方)
    ax.text(
        0.5, 1.02, # X轴中心，Y轴稍微高出图表一点点
        subtitle, 
        transform=ax.transAxes, 
        ha='center', 
        va='bottom', 
        fontsize=12, 
        color='#555555', # 使用高级的高级灰，不喧宾夺主
        fontstyle='italic' # 斜体增加学术感
    )

    ax.set_xlabel("LPIPS Distance (Lower is better preservation) →", fontsize=12)
    ax.set_ylabel("D-CLIP Score (Higher is better alignment) ↑", fontsize=12)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11, loc='best')

    if global_all_x and global_all_y:
        import numpy as np
        # 这里使用平均值
        cross_x = np.mean(global_all_x) 
        cross_y = np.mean(global_all_y)
        
        ax.axvline(x=cross_x, color='gray', linestyle='--', alpha=0.5, zorder=0)
        ax.axhline(y=cross_y, color='gray', linestyle='--', alpha=0.5, zorder=0)

        
    ax.annotate('Optimal Zone\n(High Edit, High Preserve)', xy=(0.05, 0.90), xycoords='axes fraction', 
                fontsize=11, color='green', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    
    plt.tight_layout()

    # ================= 新增：自动保存图像逻辑 =================
    try:
        import time 
        
        # 独立生成文件后缀，绝对不依赖你原来的代码逻辑
        if len(task_filter) >= 5:
            file_suffix = "All_Tasks"
        else:
            file_suffix = f"Filtered_{len(task_filter)}"

        # 生成安全的文件名：包含复杂度级别和过滤状态
        safe_complexity = str(complexity_level).replace(" ", "_").replace("/", "_")
        timestamp = int(time.time())
        base_filename = f"Pareto_{safe_complexity}_{file_suffix}_{timestamp}"
        
        save_png_path = os.path.join(results_dir, f"{base_filename}.png")
        save_pdf_path = os.path.join(results_dir, f"{base_filename}.pdf")
        
        # 导出高清PNG (dpi=400) 和 矢量PDF格式
        fig.savefig(save_png_path, dpi=400, bbox_inches='tight')
        fig.savefig(save_pdf_path, bbox_inches='tight')
        print(f"✅ 帕累托图已成功保存至: \n   - {save_png_path}\n   - {save_pdf_path}")
    except Exception as e:
        print(f"⚠️ 图像保存失败: {e}")
    # ==========================================================
    
    return fig