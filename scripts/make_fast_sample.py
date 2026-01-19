import pandas as pd
import numpy as np
import os
import joblib
import shap
import warnings

warnings.filterwarnings('ignore')

# ================= 配置路径 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ready_test.parquet')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'results', 'models', 'LightGBM.pkl')

# 输出文件路径
OUTPUT_PARQUET = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sample_with_time.parquet')
OUTPUT_NPY = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sample_with_time_shap.npy')

SAMPLE_N = 10000  # 采样 1万条

# 特征列表 (必须与模型训练时严格一致)
FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

def make_sample_and_calc_shap():
    print(f"1. 正在读取大文件: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入数据文件: {INPUT_FILE}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return

    # 1. 读取数据
    df = pd.read_parquet(INPUT_FILE)
    
    # 保留 Time/Stock 用于切分，以及特征列
    cols_to_keep = ['Time', 'time', 'Stock'] + FEATURE_COLS
    available_cols = [c for c in df.columns if c in cols_to_keep]
    df = df[available_cols]
    
    # 2. 采样
    print(f"2. 正在采样 {SAMPLE_N} 条数据...")
    if len(df) > SAMPLE_N:
        # 随机采样，并按索引排序保持物理顺序（虽然对随机森林不重要，但对时序数据习惯更好）
        df_sample = df.sample(n=SAMPLE_N, random_state=42).sort_index()
    else:
        df_sample = df
        
    # 3. 保存 Parquet
    print(f"3. 保存采样数据至 {OUTPUT_PARQUET} ...")
    df_sample.to_parquet(OUTPUT_PARQUET)
    
    # 4. 计算 SHAP (核心新增步骤)
    print("4. 正在加载模型并计算 SHAP 交互值 (慢，请稍候)...")
    
    # 加载模型
    full_model = joblib.load(MODEL_PATH)
    model = full_model.booster_ if hasattr(full_model, 'booster_') else full_model
    
    # 初始化解释器 (使用快速模式)
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    
    # 准备特征矩阵 X (排除 Time, Stock)
    X_sample = df_sample[FEATURE_COLS]
    
    # 计算交互值 [N_samples, N_features, N_features]
    shap_interaction_values = explainer.shap_interaction_values(X_sample)
    
    # 5. 保存 NPY
    print(f"5. 保存 SHAP 缓存至 {OUTPUT_NPY} ...")
    np.save(OUTPUT_NPY, shap_interaction_values)
    
    print(f"✅ 全部完成！\n数据形状: {df_sample.shape}\nSHAP形状: {shap_interaction_values.shape}")

if __name__ == '__main__':
    make_sample_and_calc_shap()