# 毕业设计：图像编辑评测与对比平台 (Image Editing Arena)

## 🗺️ 毕设全局架构蓝图 (双轨制评测平台)

```mermaid
graph TD
    %% 核心系统
    SYS["🚀 毕业设计：图像编辑评测与对比平台<br>(Image Editing Arena)"] 

    %% 三大核心模块
    SYS --> T1("🛤️ 第一轨道：本地基准沙盒<br>White-box Baselines")
    SYS --> T2("🏙️ 第二轨道：云端 SOTA 竞技场<br>Black-box API")
    SYS --> EVAL("⚖️ 客观评测流水线<br>Benchmark & Metrics")

    %% 第一轨道：底层机制 (你的重点)
    T1 --> B1["1. SDEdit<br>垫底: 纯加噪去噪"]
    T1 --> B2["2. InstructPix2Pix<br>端到端: 粗犷全局注入"]
    T1 --> B3["3. P2P + NTI<br>免训练: 精准注意力微调"]
    T1 -.-> B4["4. PnP / MasaCtrl<br>展望: 自注意力控制"]

    %% 第二轨道：商业应用 (降维对比)
    T2 --> C1["1. CosXL / SDXL-Edit<br>开源界高分指令之王"]
    T2 --> C2["2. DALL·E 3<br>商用界 LLM 意图理解天花板"]
    T2 -.-> C3["3. SmartEdit<br>前沿: MLLM 智能体架构"]

    %% 评测流水线 (裁判系统)
    EVAL --> D1[("💾 Hybrid-EditBench<br>TEDBench + MagicBrush + ReasonEdit")]
    EVAL --> D2{"🤖 机器裁判程序<br>eval_metrics.py"}
    
    %% 打分维度
    D2 --> M1["LPIPS<br>原图结构破坏率"]
    D2 --> M2["Target CLIP<br>最终目标匹配度"]
    D2 --> M3["Directional CLIP<br>编辑语义方向"]
    D2 --> M4["Latency<br>系统工程耗时"]

    %% 数据流向
    D1 -.->|"提供原图与指令"| T1
    D1 -.->|"提供原图与指令"| T2
    T1 -.->|"本地生成的图片"| D2
    T2 -.->|"云端生成的图片"| D2
    D2 ===>|"导出 CSV 数据"| CHART("📊 论文核心图表生成")

    classDef core fill:#f9f,stroke:#333,stroke-width:2px;
    classDef track fill:#bbf,stroke:#333,stroke-width:2px;
    classDef eval fill:#bfb,stroke:#333,stroke-width:2px;
    class SYS core;
    class T1,T2 track;
    class EVAL,D2 eval;