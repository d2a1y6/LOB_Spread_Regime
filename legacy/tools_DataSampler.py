"""
程序名称：tools_DataSampler.py
功能：从原始测试集中随机采样 5000 条数据并保存，供 SHAP 分析使用。
保证后续分析的数据固定，不会因随机性导致结果跳变。
"""

import pandas as pd
import os

# 全局配置
INPUT_FILE = 'ready_test.parquet'
OUTPUT_FILE = 'sample_for_shap.parquet'
SAMPLE_SIZE = 5000
SEED = 42

# 特征列 (保持一致)
FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

def run_sampling():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 确保只包含数值特征，防止读取时间戳等无关列
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available_cols]

    if len(df) > SAMPLE_SIZE:
        print(f"正在进行随机采样 (N={SAMPLE_SIZE}, Seed={SEED})...")
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        print("数据量不足采样数，保留全部数据。")
        df_sample = df
        
    print(f"正在保存至 {OUTPUT_FILE} ...")
    df_sample.to_parquet(OUTPUT_FILE)
    print("采样完成！")

if __name__ == '__main__':
    # 切换目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_sampling()