```mermaid
graph TD
    %% 核心系统
    SYS["毕业设计：图像编辑评测与对比平台<br>(Image Editing Arena)"] 

    %% 前端交互层
    SYS --> WEB["前端交互与可视化平台<br>(Web UI / 单次评测调试)"]

    %% 三大核心后备模块
    SYS --> T1("第一轨道：本地开源基准<br>(White-box Baselines)")
    SYS --> T2("第二轨道：云端前沿竞技场<br>(Black-box API)")
    SYS --> EVAL("客观自动化评测流水线<br>(Benchmark & Metrics)")

    %% 第一轨道：本地机制
    WEB -.->|"集成与参数滑块控制"| T1
    T1 --> B1["1. Stable Diffusion 1.5<br>(基准: 原生扩散生成)"]
    T1 --> B2["2. InstructPix2Pix<br>(端到端: 自然语言全局注入)"]
    T1 --> B3["3. Pix2Pix-Zero<br>(免训练: 交叉注意力精准替换)"]

    %% 第二轨道：云端应用
    WEB -.->|"接口中转与结果展示"| T2
    T2 --> C1["1. Qwen-Image-Edit-Plus<br>(国内前沿: 多模态对话编辑)"]
    T2 --> C2["2. 待扩展前沿 API 库<br>(如 CosXL / SDXL-Edit 等)"]

    %% 评测流水线：数据前置处理
    LLM["LLM 数据自动清洗工厂<br>(上下文保留 & 语义三元组对齐)"] --> D1
    
    %% 评测流水线：数据集与裁判
    EVAL --> D1[("Hybrid-EditBench 数据集<br>(按简单/复杂场景划分为五大编辑类别)")]
    EVAL --> D2{"自动化机器裁判引擎<br>(eval_metrics.py)"}
    
    %% 打分维度 (全面升级)
    D2 --> M1["LPIPS<br>(原图结构保真度)"]
    D2 --> M2["Directional CLIP (D-CLIP)<br>(支持动态路由的编辑语义方向测试)"]
    D2 --> M3["Latency<br>(云端与本地分离的计算耗时统计)"]

    %% 数据流向闭环
    D1 -.->|"分发: 图像、Instruction 与 Prompt"| T1
    D1 -.->|"分发: 图像、Instruction 与 Prompt"| T2
    T1 -.->|"本地批量生成结果"| D2
    T2 -.->|"云端批量生成结果"| D2
    D2 ===>|"聚合导出 CSV 数据"| CHART("论文核心图表生成与 Failure Case 定性分析")

    %% 样式定义
    classDef core fill:#f9f,stroke:#333,stroke-width:2px;
    classDef track fill:#bbf,stroke:#333,stroke-width:2px;
    classDef eval fill:#bfb,stroke:#333,stroke-width:2px;
    classDef frontend fill:#ffe4b5,stroke:#333,stroke-width:2px;
    classDef llm fill:#ddd,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    
    class SYS core;
    class T1,T2 track;
    class EVAL,D2,D1 eval;
    class WEB frontend;
    class LLM llm;
