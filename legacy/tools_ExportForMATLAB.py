"""
程序名称：10_ExportForMatlab.py
功能：将 SHAP 交互数据导出为 CSV，供 MATLAB 进行 3D 可视化。
"""

import pandas as pd
import numpy as np
import shap
import joblib
import os

# ================= 配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

MODEL_PATH = 'models/LightGBM.pkl'
SAMPLE_FILE = 'sample_for_shap.parquet'
CACHE_FILE = 'shap_interaction_cache.npy'
OUTPUT_CSV = 'shap_3d_data.csv'

# 指定要画 3D 图的两个特征
FEAT_X = 'Accum_Vol_Diff'   # 动力
FEAT_Y = 'Relative_Spread'  # 阻力

FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

def export_data():
    # 1. 加载数据
    print(">>> Loading Data & Model...")
    if not os.path.exists(SAMPLE_FILE) or not os.path.exists(CACHE_FILE):
        print("错误：请先运行 00_DataSampler.py 和 08_ShapAnalysis.py 生成数据和缓存。")
        return

    X_sample = pd.read_parquet(SAMPLE_FILE)
    shap_interaction = np.load(CACHE_FILE) # (N, M, M)

    # 2. 提取 X, Y, Z
    print(f">>> Extracting {FEAT_X} vs {FEAT_Y}...")
    idx_x = FEATURE_COLS.index(FEAT_X)
    idx_y = FEATURE_COLS.index(FEAT_Y)

    x_vals = X_sample[FEAT_X].values
    y_vals = X_sample[FEAT_Y].values
    
    # Z轴：取两个特征的交互值之和 (Interaction is symmetric matrix)
    # shap_interaction[:, i, j] 是 i 对 j 的影响
    # 我们取 shap_interaction[:, idx_x, idx_y] * 2 作为总交互效应
    z_vals = shap_interaction[:, idx_x, idx_y] * 2

    # 3. 极端值截尾 (Top/Bottom 1%，为了 3D 图好看)
    # 3D 图如果有一个极值点，整个面会被压得很扁，所以这里截尾稍微严一点
    def clip_series(series):
        lower = np.percentile(series, 1)
        upper = np.percentile(series, 99)
        return np.clip(series, lower, upper)

    x_clipped = clip_series(x_vals)
    y_clipped = clip_series(y_vals)
    # Z 值通常不需要 clip，保留原貌

    # 4. 保存为 CSV
    df_out = pd.DataFrame({
        'X_Accum_Vol_Diff': x_clipped,
        'Y_Relative_Spread': y_clipped,
        'Z_SHAP_Interaction': z_vals
    })

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f">>> 导出完成！文件已保存至: {OUTPUT_CSV}")
    print(">>> 现在请打开 MATLAB 运行可视化脚本。")

if __name__ == '__main__':
    export_data()