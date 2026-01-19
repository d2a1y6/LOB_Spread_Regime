"""
程序名称：09_Verify_IM_Relation.py
功能：验证 LobImbalance (I) 与 Micro_Mid_Spread (M) 的代数关系及 SHAP 交互分布
说明：
    该程序将重绘交互图。
    - 横轴：LobImbalance
    - 纵轴：Micro_Mid_Spread
    - 颜色：SHAP Interaction Value
    如果“离散价差”假设成立，图样应呈现为几条通过原点的放射状直线。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

# ==============================================================================
# 1. 配置与加载 (复用 08_ShapAnalysis.py 的路径)
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

SAMPLE_FILE = 'sample_for_shap.parquet'
CACHE_FILE = 'shap_interaction_cache.npy'
OUTPUT_DIR = 'analysis_results/08_ShapAnalysis'

# 特征列表 (必须与 08 程序完全一致以保证索引对齐)
FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

def load_data_and_shap():
    if not os.path.exists(SAMPLE_FILE) or not os.path.exists(CACHE_FILE):
        raise FileNotFoundError("错误：未找到数据或SHAP缓存。请先运行 08_ShapAnalysis.py 生成缓存。")
    
    print(f">>> 正在加载数据: {SAMPLE_FILE}")
    df = pd.read_parquet(SAMPLE_FILE)
    
    print(f">>> 正在加载SHAP缓存: {CACHE_FILE}")
    shap_values = np.load(CACHE_FILE)
    
    return df, shap_values

# ==============================================================================
# 2. 绘图逻辑
# ==============================================================================

def plot_verification(df, shap_interaction_values):
    # 1. 获取索引
    try:
        idx_I = FEATURE_COLS.index('LobImbalance')
        idx_M = FEATURE_COLS.index('Micro_Mid_Spread')
    except ValueError:
        print("错误：特征名称不匹配，请检查 FEATURE_COLS 设置。")
        return

    # 2. 提取数据
    I_val = df['LobImbalance'].values
    M_val = df['Micro_Mid_Spread'].values
    
    # 提取交互值: Shape (N, Features, Features) -> (N,)
    # 取 I 和 M 之间的交互项
    interaction_val = shap_interaction_values[:, idx_I, idx_M]

    # 3. 数据清洗 (为了视觉清晰，裁剪掉极端的离群点，例如前后 1%)
    # 注意：保留这一步是为了防止极个别噪点压缩了颜色条的范围
    mask_I = (I_val >= np.percentile(I_val, 0.5)) & (I_val <= np.percentile(I_val, 99.5))
    mask_M = (M_val >= np.percentile(M_val, 0.5)) & (M_val <= np.percentile(M_val, 99.5))
    mask = mask_I & mask_M
    
    X = I_val[mask]
    Y = M_val[mask]
    C = interaction_val[mask]

    # 4. 绘制散点图
    print(f">>> 正在绘图 (样本数: {len(X)})...")
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # 使用 seismic (红蓝) 色谱，0值居中为白色/浅色
    # 调整 vmin/vmax 让颜色对比更强烈 (使用稳健的百分位数范围)
    c_max = np.percentile(np.abs(C), 98)
    
    sc = plt.scatter(X, Y, c=C, cmap='seismic', 
                     s=1, alpha=0.6, 
                     vmin=-c_max, vmax=c_max) # s=1 保证点足够小，看清结构
    
    # 5. 添加辅助元素
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('SHAP Interaction Value (Color)', rotation=270, labelpad=20)
    
    # 标签与标题
    plt.xlabel('Feature: LobImbalance (I)', fontsize=12)
    plt.ylabel('Feature: Micro_Mid_Spread (M)', fontsize=12)
    plt.title('Verification: Physical Relationship vs Model Interaction\n(Is M = -S/2 * I ?)', fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.4)
    
    # 6. 保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    filename = '99_Verification_Lob_vs_MicroSpread.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    print(f">>> 绘图完成: {save_path}")
    # plt.show() # 如果在服务器运行请注释此行

if __name__ == '__main__':
    try:
        df, shap_vals = load_data_and_shap()
        plot_verification(df, shap_vals)
    except Exception as e:
        print(f"运行出错: {e}")