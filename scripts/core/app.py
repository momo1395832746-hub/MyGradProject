import gradio as gr
from editors import InstructPix2PixEditor, Pix2PixZeroEditor, SDEditEditor

editors = {
    "SDEdit (基准线)": SDEditEditor(),
    "InstructPix2Pix": InstructPix2PixEditor(),
    "Pix2Pix-Zero": Pix2PixZeroEditor()
}

def process_edit(input_image, prompt, source_prompt, model_choice, 
                 steps, txt_cfg, img_cfg, cag_amount):
    
    if input_image is None or not prompt:
        return None
        
    # 卸载不需要的模型
    for name, ed in editors.items():
        if name != model_choice and getattr(ed, 'model', None) is not None:
            print(f"[资源调度] 正在卸载未使用的模型: {name}")
            ed.clear_vram()
            
    editor = editors.get(model_choice)
    if not editor:
        print(f"错误: 未找到对应的模型 {model_choice}")
        return None
        
    # 适配器模式：调用底层统一接口
    result_img = editor.edit_image(
        image=input_image, 
        prompt=prompt,
        source_prompt=source_prompt,         
        num_inference_steps=steps,           
        guidance_scale=txt_cfg,              
        image_guidance_scale=img_cfg,        
        cross_attention_guidance_amount=cag_amount     
    )
    
    return result_img

def update_ui(model_choice):
    """
    根据选择的模型，动态显示或隐藏对应的参数滑块和输入框
    返回的是 gr.update() 对象的列表，顺序与 outputs 严格对应
    """
    if model_choice == "InstructPix2Pix":
        return [
            gr.update(visible=False), # 源提示词
            gr.update(visible=True, value=20),  # 推理步数
            gr.update(visible=True),  # 文本CFG
            gr.update(visible=True),  # 图像CFG
            gr.update(visible=False), # CAG引导强度
        ]
    elif model_choice == "Pix2Pix-Zero":
        return [
            gr.update(visible=True),  
            gr.update(visible=True, value=50),  
            gr.update(visible=True, value=5.0), 
            gr.update(visible=False),
            gr.update(visible=True, value=0.1),  
        ]
    elif model_choice == "SDEdit(基准线)":
        return [
            gr.update(visible=False), 
            gr.update(visible=True, value=50),  
            gr.update(visible=True),  
            gr.update(visible=False), 
            gr.update(visible=False), 
        ]

# 构建 UI 界面
with gr.Blocks(title="指令驱动图像编辑方法比较研究平台") as demo:
    gr.Markdown("## 基于扩散模型的指令驱动图像编辑系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传原图", type="pil")
            prompt_input = gr.Textbox(label="编辑指令 / 目标提示词 (Target Prompt)", placeholder="例如：turn him into a cyborg")
            
            # 默认隐藏，只有切到 P2P/NTI 才会弹出来
            source_prompt_input = gr.Textbox(label="源提示词 (Source Prompt)", placeholder="例如：a statue of a man", visible=False)
            
            # 模型选择下拉框
            model_dropdown = gr.Dropdown(
                choices=list(editors.keys()), 
                value="SDEdit (基准线)", 
                label="选择算法模型"
            )
            
            # 高级参数面板
            with gr.Accordion("高级参数设置 (Advanced Settings)", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(minimum=10, maximum=100, value=20, step=1, label="推理步数 (Steps)")
                    txt_cfg_slider = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="文本引导强度 (Text CFG)")
                with gr.Row():
                    # 给专属参数加上 visible=False 初始状态
                    img_cfg_slider = gr.Slider(minimum=1.0, maximum=3.0, value=1.5, step=0.1, label="图像引导强度 (Image CFG)")
                    cag_amount_slider = gr.Slider(minimum=0.0, maximum=0.5, value=0.15, step=0.01, label="注意力引导强度 (CAG Amount)", visible=False)
                
            submit_btn = gr.Button("开始编辑", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="编辑结果")
            
    model_dropdown.change(
        fn=update_ui,
        inputs=[model_dropdown],
        outputs=[
            source_prompt_input,
            steps_slider,
            txt_cfg_slider,
            img_cfg_slider,
            cag_amount_slider
        ]
    )
            
    # 绑定生成按钮
    submit_btn.click(
        fn=process_edit,
        inputs=[
            input_image, prompt_input, source_prompt_input, model_dropdown, 
            steps_slider, txt_cfg_slider, img_cfg_slider, cag_amount_slider
        ],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=6006)