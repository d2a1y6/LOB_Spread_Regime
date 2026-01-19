"""
程序名称：09_HypothesisVerification.py - 微观结构假说验证 (LobImbalance vs Spread)

1. 功能概述：
   结合 01_GetDataFromDB 的数据结构，执行 Memo0113 中的三个验证任务：
   - Task A: 验证宽价差下 Imbalance 的数据稀疏性。
   - Task B: 验证窄价差下"陷阱区"的收益率反转 (PnL Analysis)。
   - Task C: 验证不同价差环境下特征重要性的漂移。

2. 数据依赖：
   - sample_for_shap.parquet (包含 DolphinDB 生成的特征)
   - models/LightGBM.pkl (可选，用于 Task C)

3. 特殊处理：
   - 自动从 Relative_Spread(bp) 逆向推导 Spread_Ticks。
   - 自动处理 DolphinDB 脚本中的特征缩放 (Scaling)。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.inspection import permutation_importance

# 配置绘图
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 配置与工具函数
# ==============================================================================
SAMPLE_FILE = 'sample_for_shap.parquet'
MODEL_PATH = 'models/LightGBM.pkl'
TICK_SIZE = 0.01  # 沪市主板通常为 0.01

def load_and_process_data():
    """加载数据并恢复物理含义"""
    if not os.path.exists(SAMPLE_FILE):
        raise FileNotFoundError(f"未找到数据文件: {SAMPLE_FILE}")
    
    print(f">>> 正在加载数据: {SAMPLE_FILE} ...")
    df = pd.read_parquet(SAMPLE_FILE)
    
    # [关键步骤] 逆向工程：从 Relative_Spread 恢复 Spread_Ticks
    # 逻辑：Relative_Spread = (Ask-Bid)/Mid * 10000 
    # 因此：RawSpread = Relative_Spread / 10000 * MidPrice
    #      Ticks = RawSpread / 0.01
    
    # 确保 MidPrice 存在
    if 'MidPrice' not in df.columns:
        # 如果样本中没有 MidPrice，尝试用 MicroPrice 或其他近似，或报错
        # 这里假设 DolphinDB 的输出包含了 MidPrice
        raise ValueError("数据中缺少 'MidPrice' 列，无法计算 Tick Spread。")

    print(">>> 正在计算离散价差 (Spread Ticks)...")
    raw_spread = (df['Relative_Spread'] / 10000.0) * df['MidPrice']
    df['Spread_Ticks'] = (raw_spread / TICK_SIZE).round().astype(int)
    
    # 简单清洗：Spread 至少为 1
    df['Spread_Ticks'] = df['Spread_Ticks'].clip(lower=1)
    
    # [关键步骤] 构造 Target (未来收益)
    # DolphinDB 脚本只计算了 PastReturn 
    # 我们假设数据是按时间排序的，用下一行的 PastReturn 作为当前行的 FutureReturn
    # 注意：这在跨股票时会出错，严谨做法需按 Stock 分组 shift
    # 这里简化处理，或者假设数据中已有 Label 字段
    if 'Label' in df.columns:
        df['Next_Ret'] = df['Label']
    else:
        print(">>> 警告：未找到 'Label' 列，正在使用 shift(-1) 构造临时 Target...")
        df['Next_Ret'] = df.groupby('Stock')['PastReturn'].shift(-1)
        df.dropna(subset=['Next_Ret'], inplace=True)
        
    return df

# ==============================================================================
# Task A: 数据分布检验 (Sparsity Check)
# ==============================================================================
def run_task_a(df):
    print("\n=== [Task A] 数据分布检验 ===")
    
    # 分组：Spread=1, Spread=2, Spread>=3
    conditions = [
        df['Spread_Ticks'] == 1,
        df['Spread_Ticks'] == 2,
        df['Spread_Ticks'] >= 3
    ]
    choices = ['Spread=1', 'Spread=2', 'Spread>=3']
    df['Spread_Group'] = np.select(conditions, choices, default='Other')
    
    # 1. 打印统计量
    print("各组 LobImbalance > 0.8 (极端值) 的样本占比:")
    for group in choices:
        sub = df[df['Spread_Group'] == group]
        if len(sub) == 0: continue
        extreme_ratio = (sub['LobImbalance'].abs() > 0.8).mean()
        print(f"  - {group}: {extreme_ratio:.2%} (样本数: {len(sub)})")
        
    # 2. 绘图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='LobImbalance', hue='Spread_Group', 
                fill=True, common_norm=False, palette='viridis', alpha=0.3)
    plt.title('LobImbalance Distribution Conditional on Spread')
    plt.xlabel('LobImbalance (-1 to 1)')
    plt.xlim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig('analysis_results/09_TaskA_Distribution.png')
    print("  -> 图表已保存: analysis_results/09_TaskA_Distribution.png")

# ==============================================================================
# Task B: PnL / 事件研究 (Event Study)
# ==============================================================================
def run_task_b(df):
    print("\n=== [Task B] 收益率反转检验 (PnL Event Study) ===")
    
    # 定义三种微观状态
    # 1. 陷阱区 (Trap): Spread=1, Imbalance > 0.8 (看似买压大，Hypothesis A 预测下跌)
    mask_trap = (df['Spread_Ticks'] == 1) & (df['LobImbalance'] > 0.8)
    
    # 2. 动量区 (Momentum): Spread=1, 0.3 < Imbalance < 0.6 (买压适中，预测上涨)
    mask_mom = (df['Spread_Ticks'] == 1) & (df['LobImbalance'] > 0.3) & (df['LobImbalance'] < 0.6)
    
    # 3. 宽价差区 (Wide): Spread>=3, Imbalance > 0.8 (真实买压，预测上涨)
    mask_wide = (df['Spread_Ticks'] >= 3) & (df['LobImbalance'] > 0.8)
    
    stats = []
    for name, mask in [('Trap (s=1, I>0.8)', mask_trap), 
                       ('Mom (s=1, 0.3<I<0.6)', mask_mom), 
                       ('Wide (s>=3, I>0.8)', mask_wide)]:
        avg_ret = df.loc[mask, 'Next_Ret'].mean()
        count = mask.sum()
        stats.append({'Regime': name, 'Avg_Next_Ret': avg_ret, 'Count': count})
        
    res_df = pd.DataFrame(stats)
    print(res_df)
    
    # 绘图
    plt.figure(figsize=(8, 5))
    colors = ['red' if v < 0 else 'green' for v in res_df['Avg_Next_Ret']]
    sns.barplot(x='Regime', y='Avg_Next_Ret', data=res_df, palette=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Average Future Return by Microstructure Regime')
    plt.ylabel('Avg Next Return (Scaled)')
    plt.tight_layout()
    plt.savefig('analysis_results/09_TaskB_PnL.png')
    print("  -> 图表已保存: analysis_results/09_TaskB_PnL.png")

# ==============================================================================
# Task C: 特征重要性漂移 (Feature Drift)
# ==============================================================================
def run_task_c(df):
    if not os.path.exists(MODEL_PATH):
        print("\n[Task C] Skip: 未找到模型文件。")
        return

    print("\n=== [Task C] 特征重要性漂移验证 ===")
    model = joblib.load(MODEL_PATH)
    
    # 获取模型使用的特征列
    # 注意：需确保 df 中包含模型训练时的所有特征
    try:
        model_features = model.feature_name()
    except:
        # 如果无法直接获取，使用默认特征列表 (需与 08_ShapAnalysis 保持一致)
        model_features = [
            'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
            'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
            'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
        ]
    
    # 确保特征在 df 中
    valid_features = [f for f in model_features if f in df.columns]
    
    importance_data = []
    groups = ['Spread=1', 'Spread>=3']
    
    for g in groups:
        if g == 'Spread=1':
            sub_df = df[df['Spread_Ticks'] == 1]
        else:
            sub_df = df[df['Spread_Ticks'] >= 3]
            
        if len(sub_df) < 500: continue
        
        # 采样以加速计算
        sample_size = min(2000, len(sub_df))
        sub_sample = sub_df.sample(sample_size, random_state=42)
        
        X = sub_sample[valid_features]
        y = sub_sample['Next_Ret'] # 使用构造的 Target
        
        # 计算 Permutation Importance
        r = permutation_importance(model, X, y, n_repeats=3, random_state=42)
        
        # 记录 LobImbalance 和 Accum_Vol_Diff 的重要性
        for feat in ['LobImbalance', 'Accum_Vol_Diff']:
            if feat in valid_features:
                idx = valid_features.index(feat)
                importance_data.append({
                    'Group': g,
                    'Feature': feat,
                    'Importance': r.importances_mean[idx]
                })

    if not importance_data:
        print("数据不足，无法计算 Task C")
        return

    imp_df = pd.DataFrame(importance_data)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Group', y='Importance', hue='Feature', data=imp_df)
    plt.title('Feature Importance Shift: Narrow vs Wide Spread')
    plt.tight_layout()
    plt.savefig('analysis_results/09_TaskC_Importance.png')
    print("  -> 图表已保存: analysis_results/09_TaskC_Importance.png")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    # 确保输出目录存在
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
        
    # 1. 加载数据
    try:
        df = load_and_process_data()
        
        # 2. 执行验证任务
        run_task_a(df)
        run_task_b(df)
        run_task_c(df)
        
        print("\n====== 所有验证任务已完成 ======")
        
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()