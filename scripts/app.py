import gradio as gr
from editors import InstructPix2PixEditor

# 1. 初始化模型管理器字典
editors = {
    "InstructPix2Pix": InstructPix2PixEditor()
    # 未来这里会添加 "Prompt-to-Prompt": PromptToPromptEditor() 等
}

# 2. 定义前端触发的核心处理函数
def process_edit(image, prompt, model_choice, steps, cfg_scale, image_cfg):
    if image is None or not prompt:
        return None
    
    # 获取选中的模型实例
    editor = editors.get(model_choice)
    if not editor:
        return None
        
    # 调用统一的 edit_image 接口 (后端暂时忽略滑块传来的 steps/cfg_scale)
    result_img = editor.edit_image(
        image=image, 
        prompt=prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        image_cfg=image_cfg
    )
    return result_img

# 3. 构建 UI 界面
with gr.Blocks(title="毕业设计：指令驱动图像编辑平台") as demo:
    gr.Markdown("## 基于扩散模型的指令驱动图像编辑系统")
    
    with gr.Row():
        # 左侧控制台
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传原图", type="pil")
            prompt_input = gr.Textbox(label="编辑指令 (Prompt)", placeholder="例如：turn him into a cyborg")
            
            model_dropdown = gr.Dropdown(
                choices=["InstructPix2Pix"], 
                value="InstructPix2Pix", 
                label="选择算法模型"
            )
            
            with gr.Accordion("高级参数调节 (当前阶段不可用)", open=True):
                steps_slider = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="推理步数 (Steps)")
                cfg_slider = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="文本引导权重 (CFG Scale)")
                img_cfg_slider = gr.Slider(minimum=1.0, maximum=3.0, value=1.5, step=0.1, label="原图保持权重 (Image Guidance)")
                
            submit_btn = gr.Button("开始编辑", variant="primary")

        # 右侧结果展示
        with gr.Column(scale=1):
            output_image = gr.Image(label="编辑结果")
            
    # 绑定点击事件：将输入组件的值按顺序传递给 process_edit 函数，结果输出到 output_image
    submit_btn.click(
        fn=process_edit,
        inputs=[input_image, prompt_input, model_dropdown, steps_slider, cfg_slider, img_cfg_slider],
        outputs=output_image
    )

# 4. 启动服务
if __name__ == "__main__":
    # share=True 生成公网链接，方便你在本地浏览器访问 AutoDL 的服务
    demo.launch(share=False, server_name="0.0.0.0", server_port=6006)