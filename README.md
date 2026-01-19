# LOB-Spread-Regime: High-Frequency Direction Prediction

本项目研究 **A股微观结构中的非线性状态机制**。
核心发现：挂单不平衡（LobImbalance）对未来价格方向的预测能力，会随着买卖价差（Spread）的离散变化而发生结构性反转。

## 项目结构

```text
LOB-Spread-Regime/
├── data/                    # 【本地数据】(Git ignored)
│   ├── raw/                 # 原始高频 Tick/Snapshot 数据
│   └── processed/           # 清洗后的特征矩阵
│
├── docs/                    # 【科研笔记】思路、提纲、AI讨论记录
│   ├── brainstorming/       # 各种思路笔记
│   └── resources/           # 外部资料
│
├── legacy/                  # 【归档】旧版代码与工具脚本
│   ├── 01_GetDataFromDB_sh.dos       # 沪市Level2高频数据特征提取
│   ├── 01_GetDataFromDB_sz.dos       # 深市Level2高频数据特征提取
│   ├── 02_ConcatDataFromDB.py        # CSV合并排序为Parquet
│   ├── 03_PreprocessDataForTrain.py  # 数据清洗+标签生成+切分标准化
│   ├── 04_ClassicMLModels.py         # 经典ML模型对比训练
│   ├── 05_ResNet_GPU.py              # ResNet-MLP深度学习模型
│   ├── 06_AttentionGRU_GPU.py        # Attention-GRU时序模型
│   ├── 07_ResNet_Attention_GRU_GPU.py # ResNet+GRU+Attention混合模型
│   ├── 08_ResNet_AttentionGRU_Improved.py # 改进版混合模型
│   ├── 09_ShapAnalysis.py            # SHAP交互值计算与可视化
│   ├── 10_TreeDiagram.py             # LightGBM树结构可视化
│   ├── tools_3DDisplay.m             # MATLAB 3D SHAP曲面图
│   ├── tools_CheckCUDA.py            # CUDA环境检测
│   ├── tools_DataSampler.py          # 固定样本采样
│   ├── tools_ExportForMATLAB.py      # 导出SHAP数据给MATLAB
│   ├── tools_HypothesisVerification.py # 微观结构假说验证
│   ├── tools_LGBMOptuna.py           # LightGBM超参数调优
│   ├── tools_ModelBacktest.py        # 通用模型回测工具
│   ├── tools_ModelDefinitions.py     # PyTorch模型结构定义
│   ├── tools_PrintStockIDs.py        # 提取HS300成分股代码
│   ├── tools_TestSequential.py       # 查看Parquet数据
│   └── tools_Verify_IM_Relation.py   # 验证LobImbalance与价差关系
│
├── notebooks/               # 【实验台】当前的分析与可视化
│
├── results/                 # 【产出】图表与模型
│   ├── figures/             # 存放生成的 SHAP 图、机制解释图
│   └── models/              # 训练好的模型文件 (.pkl, .txt)
│
├── scripts/                 # 【脚本】ETL与批处理
│   └── etl/                 #  源数据获取与处理
│
├── src/                     # 【核心库】可复用的 Python 模块
│   ├── __init__.py
│   ├── features.py          # 特征工程 (LobImbalance, MicroSpread 等计算)
│   ├── models.py            # 模型训练与评估逻辑
│   ├── visualization.py     # 专用绘图函数
│   └── utils.py             # 通用工具 (路径处理, 日志)
│
├── .gitignore               # 忽略规则 (Data, caches)
├── README.md                # 项目说明
└── requirements.txt         # 依赖环境
```
