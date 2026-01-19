"""
程序名称：通用模型回测工具 (Universal Backtester)
功能描述：
    1. 扫描 models/ 目录下的 .pkl (Sklearn) 和 .pth (PyTorch) 模型。
    2. 从 model_definitions.py 动态加载模型结构。
    3. 自动识别模型类型（ResNet/GRU/LGBM），适配输入数据格式（2D表格/3D序列）。
    4. 输出 AUC、Accuracy 等核心指标，并生成详细报告。

输入：
    - 数据：ready_test.parquet
    - 模型：models/ 文件夹下的所有模型文件。
输出：
    - 屏幕打印评估指标。
    - models/ 目录下生成 _report.txt 和 backtest_summary.csv。
"""

import pandas as pd
import numpy as np
import joblib
import os
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

# 导入自定义模型结构
from tools_ModelDefinitions import ResNetMLP, GRUWithAttention, CNN_GRU_Model

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置
# ==============================================================================
TEST_DATA_PATH = 'ready_test.parquet'
MODEL_DIR = 'models'     
SEQ_LEN = 20  # 序列模型的时间窗口
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 功能函数
# ==============================================================================

def load_test_data():
    """加载测试数据，排除非特征列"""
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"找不到数据: {TEST_DATA_PATH}")
    
    df = pd.read_parquet(TEST_DATA_PATH)
    exclude = ['Stock', 'Time', 'Label', 'Date_Only', 'Date_Group']
    feats = [c for c in df.columns if c not in exclude]
    print(f"[Data] 已加载测试集: {df.shape}, 特征数: {len(feats)}")
    return df, feats

def predict_pytorch(model_path, df, feature_cols):
    """
    加载 .pth 模型进行预测。
    处理逻辑：
    1. 识别模型架构 (ResNet/GRU)。
    2. 加载 state_dict。
    3. 数据预处理 (ResNet: 2D, GRU: 3D)。
    4. 批量推理 (ResNet 需手动 Sigmoid)。
    """
    model_name = os.path.basename(model_path).lower()
    input_dim = len(feature_cols)
    y_true = df['Label'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    
    # --- 1. 实例化模型 ---
    is_seq = False
    is_logit_output = False # 标记是否需要手动 Sigmoid
    
    try:
        if 'resnet' in model_name:
            model = ResNetMLP(input_dim=input_dim).to(DEVICE)
            is_seq = False
            is_logit_output = True # ResNetMLP 输出 Logits
            
        elif 'cnn' in model_name:
            model = CNN_GRU_Model(input_dim=input_dim).to(DEVICE)
            is_seq = True
            
        elif 'gru' in model_name:
            # 默认参数需与训练时一致 (hidden=64)
            model = GRUWithAttention(input_dim=input_dim, hidden_dim=64).to(DEVICE)
            is_seq = True
            
        else:
            print(f"    [Skip] 无法识别的模型架构: {model_name}")
            return None, None

        # --- 2. 加载权重 ---
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
    except Exception as e:
        print(f"    [Error] 模型加载失败: {e}")
        return None, None
    
    # --- 3. 推理循环 ---
    preds = []
    targets = []
    batch_size = 4096
    
    with torch.no_grad():
        if is_seq:
            # 序列数据构造 (简单滑动窗口)
            data_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y_true, dtype=torch.float32)
            X_seq = data_tensor.unfold(0, SEQ_LEN, 1).transpose(1, 2)
            y_seq = y_tensor[SEQ_LEN-1:]
            
            total = X_seq.shape[0]
            for i in range(0, total, batch_size):
                batch_X = X_seq[i:i+batch_size].to(DEVICE)
                out = model(batch_X) # 这里的模型自带 Sigmoid
                preds.extend(out.cpu().numpy().flatten())
                targets.extend(y_seq[i:i+batch_size].numpy())
        else:
            # 表格数据
            data_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            for i in range(0, len(data_tensor), batch_size):
                batch_X = data_tensor[i:i+batch_size].to(DEVICE)
                out = model(batch_X)
                
                # 关键修复：如果是 ResNet (输出 Logits)，手动加 Sigmoid
                if is_logit_output:
                    out = torch.sigmoid(out)
                    
                preds.extend(out.cpu().numpy().flatten())
                targets.extend(y_true[i:i+batch_size])
                
    return np.array(targets), np.array(preds)

def predict_sklearn(model_path, df, feature_cols):
    """加载 .pkl 模型 (LightGBM/RF) 进行预测"""
    clf = joblib.load(model_path)
    X = df[feature_cols]
    y = df['Label']
    
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X)[:, 1]
    else:
        y_prob = clf.predict(X)
        
    return y, y_prob

# ==============================================================================
# 主程序
# ==============================================================================
def run_backtest():
    if not os.path.exists(MODEL_DIR):
        print(f"[Error] 文件夹不存在: {MODEL_DIR}")
        return
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.pkl', '.pth'))]
    if not model_files:
        print("[Warn] 未发现模型文件")
        return
        
    df, feats = load_test_data()
    print(f"\n>>> 开始回测 {len(model_files)} 个模型...\n")
    
    results = []

    for f_name in model_files:
        model_path = os.path.join(MODEL_DIR, f_name)
        print(f"--- Testing: {f_name} ---")
        
        try:
            if f_name.endswith('.pth'):
                y_true, y_prob = predict_pytorch(model_path, df, feats)
                if y_true is None: continue
            else:
                y_true, y_prob = predict_sklearn(model_path, df, feats)
            
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = {
                'Model': f_name,
                'AUC': roc_auc_score(y_true, y_prob),
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1': f1_score(y_true, y_pred)
            }
            
            print(f"    AUC: {metrics['AUC']:.4f} | Acc: {metrics['Accuracy']:.4f}")
            results.append(metrics)
            
            # 保存单模型报告
            with open(os.path.join(MODEL_DIR, f"{f_name}_report.txt"), "w") as f:
                f.write(f"Model: {f_name}\nMetrics: {metrics}\n\n")
                f.write(classification_report(y_true, y_pred, digits=4))

        except Exception as e:
            print(f"    [Error] 测试中断: {e}")

    # 保存总榜单
    if results:
        res_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
        print("\n" + "="*50)
        print("FINAL LEADERBOARD")
        print("="*50)
        print(res_df[['Model', 'AUC', 'Accuracy', 'F1']].to_string(index=False, float_format="%.4f"))
        res_df.to_csv(os.path.join(MODEL_DIR, 'backtest_summary.csv'), index=False)

if __name__ == '__main__':
    run_backtest()