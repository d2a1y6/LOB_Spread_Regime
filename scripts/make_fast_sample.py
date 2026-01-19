"""
功能：从全量测试集中采样数据，并基于 LightGBM 模型计算 SHAP 交互值。
输入：
    - 数据文件: data/processed/ready_test.parquet
    - 模型文件: results/models/LightGBM.pkl
输出：
    - 采样数据: data/processed/sample_with_time.parquet
    - SHAP矩阵: data/processed/sample_with_time_shap.npy
逻辑：读取全量数据 -> 按设定数量随机采样 -> 保存采样文件 -> 加载模型构造解释器 -> 计算并保存 SHAP 交互值。
"""

import pandas as pd
import numpy as np
import os
import joblib
import shap
import warnings

warnings.filterwarnings('ignore')

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ready_test.parquet')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'results', 'models', 'LightGBM.pkl')
OUTPUT_PARQUET = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sample_with_time.parquet')
OUTPUT_NPY = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sample_with_time_shap.npy')

SAMPLE_N = 10000 
FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

def make_sample_and_calc_shap():
    """
    主函数：执行采样与 SHAP 计算流程。
    输入：无（读取全局配置路径）
    输出：无（生成文件到磁盘）
    """
    if not os.path.exists(INPUT_FILE) or not os.path.exists(MODEL_PATH):
        print(f"错误：输入文件或模型文件缺失。")
        return

    print(f"正在读取数据: {os.path.basename(INPUT_FILE)} ...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 筛选列
    cols_to_keep = ['Time', 'time', 'Stock'] + FEATURE_COLS
    df = df[[c for c in df.columns if c in cols_to_keep]]
    
    # 采样
    print(f"正在采样 {SAMPLE_N} 条数据并保存...")
    if len(df) > SAMPLE_N:
        df_sample = df.sample(n=SAMPLE_N, random_state=42).sort_index()
    else:
        df_sample = df
    df_sample.to_parquet(OUTPUT_PARQUET)
    
    # 计算 SHAP
    print("正在加载模型并计算 SHAP 交互值...")
    full_model = joblib.load(MODEL_PATH)
    model = full_model.booster_ if hasattr(full_model, 'booster_') else full_model
    
    # 使用 TreeExplainer 的 path-dependent 扰动模式
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    X_sample = df_sample[FEATURE_COLS]
    shap_interaction_values = explainer.shap_interaction_values(X_sample)
    
    np.save(OUTPUT_NPY, shap_interaction_values)
    
    print(f"完成。采样形状: {df_sample.shape}, SHAP形状: {shap_interaction_values.shape}")

if __name__ == '__main__':
    make_sample_and_calc_shap()