"""
程序名称：09_ShapAnalysis.py - SHAP 交互值计算与可视化

1. 功能概述：
   本程序基于训练好的 LightGBM 模型，通过 SHAP 交互值（Interaction Values）挖掘特征间的
   非线性耦合关系。支持结果缓存以加速后续运行，并自动绘制交互强度最高的特征组合图谱。

2. 输入数据：
   - 模型文件：'models/LightGBM.pkl'
   - 数据文件：'sample_for_shap.parquet' (由 tools_DataSampler.py 生成的固定样本)
   - 缓存文件：'shap_interaction_cache.npy' (自动生成，用于加速)

3. 输出结果：
   - 图片：'analysis_results/08_ShapAnalysis/02_interaction_*.png'

4. 处理逻辑：
   1. 加载模型与数据。
   2. 检查是否存在 SHAP 缓存：若有则读取，若无则计算并保存。
   3. 统计全矩阵交互强度，筛选 Top 5 特征对。
   4. 对 Top 5 特征对绘制去极端值的交互依赖图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全局配置
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

MODEL_PATH = 'models/LightGBM.pkl'
SAMPLE_FILE = 'sample_for_shap.parquet'
CACHE_FILE = 'shap_interaction_cache.npy'  # SHAP缓存文件路径
OUTPUT_DIR = 'analysis_results/08_ShapAnalysis'

FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 1. 核心功能函数
# ==============================================================================

def load_data():
    """
    功能：加载预采样的测试数据。
    输入：无 (读取全局配置的 SAMPLE_FILE)。
    输出：Pandas DataFrame。
    """
    if not os.path.exists(SAMPLE_FILE):
        raise FileNotFoundError(f"缺失文件: {SAMPLE_FILE}，请先运行 00_DataSampler.py。")
    return pd.read_parquet(SAMPLE_FILE)

def get_shap_interactions(model, X_sample):
    """
    功能：获取 SHAP 交互值矩阵（含缓存机制）。
    输入：模型对象, 样本数据 X_sample。
    输出：交互值矩阵 (Samples, Features, Features)。
    逻辑：
        1. 检查本地是否存在 .npy 缓存文件。
        2. 若存在，直接加载（毫秒级）。
        3. 若不存在，调用 TreeExplainer 计算（耗时），并保存至本地供下次使用。
    """
    if os.path.exists(CACHE_FILE):
        print(f">>> 发现缓存文件 {CACHE_FILE}，正在快速加载...")
        return np.load(CACHE_FILE)
    
    print(">>> 未找到缓存，正在计算 SHAP Interaction Values (首次运行较慢)...")
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X_sample)
    
    print(f">>> 计算完成，正在保存至 {CACHE_FILE} ...")
    np.save(CACHE_FILE, shap_interaction)
    return shap_interaction

def get_top_interactions(shap_interaction_values, top_n=5):
    """
    功能：计算并筛选交互强度最高的特征对。
    输入：SHAP 交互矩阵, 筛选数量 N。
    输出：Top N 列表 [(特征A, 特征B, 强度分), ...]。
    逻辑：
        1. 对矩阵取绝对值并求均值，得到特征间的平均交互强度。
        2. 忽略主效应（对角线），仅提取下三角矩阵的交互对。
        3. 按强度降序排序。
    """
    mean_interaction = np.abs(shap_interaction_values).mean(0)
    np.fill_diagonal(mean_interaction, 0)
    
    interactions = []
    n_feats = mean_interaction.shape[0]
    
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            score = mean_interaction[i, j] + mean_interaction[j, i]
            interactions.append((FEATURE_COLS[i], FEATURE_COLS[j], score))
            
    interactions.sort(key=lambda x: x[2], reverse=True)
    return interactions[:top_n]

def plot_interaction_scatter(shap_interaction, X_sample, feature_x, feature_color, filename):
    """
    功能：绘制去除极端值的交互依赖图。
    输入：SHAP 矩阵, 样本数据, X轴特征名, 颜色特征名, 输出文件名。
    输出：无 (保存图片)。
    逻辑：
        1. 计算 2.5% 和 97.5% 分位数，生成过滤掩码以剔除极端值。
        2. 使用 shap.dependence_plot 绘制纯净的散点图。
        3. 保存至指定目录。
    """
    # 数据截尾 (Trim 2.5% outliers)
    x_vals = X_sample[feature_x].values
    c_vals = X_sample[feature_color].values
    
    lower_x, upper_x = np.percentile(x_vals, [2.5, 97.5])
    lower_c, upper_c = np.percentile(c_vals, [2.5, 97.5])
    
    mask = (x_vals >= lower_x) & (x_vals <= upper_x) & \
           (c_vals >= lower_c) & (c_vals <= upper_c)
    
    X_filtered = X_sample.iloc[mask]
    shap_filtered = shap_interaction[mask]
    
    # 绘图
    plt.figure(figsize=(10, 7))
    shap.dependence_plot(
        (feature_x, feature_color),
        shap_filtered, 
        X_filtered,
        display_features=X_filtered,
        show=False,
        alpha=0.6,
        dot_size=20
    )
    
    plt.title(f"Interaction: {feature_x} vs {feature_color}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    -> 已保存: {filename}")

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 1. 资源加载
    X_sample = load_data()
    if not os.path.exists(MODEL_PATH):
        print("错误：未找到模型文件。")
        exit()
    model = joblib.load(MODEL_PATH)
    
    # 2. SHAP 计算 (含缓存)
    shap_interaction = get_shap_interactions(model, X_sample)
    
    # 3. 挖掘 Top 交互
    print(">>> 正在挖掘 Top 5 交互关系...")
    top_interactions = get_top_interactions(shap_interaction, top_n=5)
    
    # 强制包含 Accum_Vol_Diff vs Relative_Spread
    target = ('Accum_Vol_Diff', 'Relative_Spread')
    if not any((t[0]==target[0] and t[1]==target[1]) for t in top_interactions):
         top_interactions.insert(0, (target[0], target[1], 0.0))
    
    for k, (fa, fb, score) in enumerate(top_interactions):
        print(f"    {k+1}. {fa} <--> {fb} (Score: {score:.4f})")
    
    # 4. 绘图
    print(">>> 正在绘制交互依赖图...")
    for i, (feat_x, feat_color, _) in enumerate(top_interactions):
        fname = f"02_interaction_{i+1}_{feat_x}_vs_{feat_color}.png"
        plot_interaction_scatter(shap_interaction, X_sample, feat_x, feat_color, fname)

    print(f"\n>>> 分析完成。结果已保存至 {OUTPUT_DIR}")