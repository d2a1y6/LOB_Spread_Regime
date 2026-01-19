"""
程序名称：基于海量高频数据的买卖方向预测模型对比 (Big Data Edition)

1. 程序概述：
   本程序旨在对比分析多种机器学习模型（线性与非线性）在海量高频交易限价订单簿(LOB)数据上的预测性能。
   任务目标是利用微观结构特征（如深度失衡、流动性压力、波动率等）预测下个3秒的价格变动方向（涨/跌）。
   程序集成了 Parquet 数据加载、多模型批量训练（针对大数据优化）、评估指标计算及可视化分析功能。

2. 输入数据：
   - 训练集文件：'ready_train.parquet' (已包含清洗、标签生成、时间切分及标准化的数据)
   - 测试集文件：'ready_test.parquet'
   - 特征列(X)：13个核心微观结构特征，包括：
     Accum_Vol_Diff, VolumeMax, VolumeAll, Immediacy, Depth_Change,
     LobImbalance, DeepLobImbalance, Relative_Spread, Micro_Mid_Spread,
     PastReturn, Lambda, Volatility, AutoCov
   - 标签列(y)：'Label' (1: 价格上涨, 0: 价格下跌)。

3. 包含模型：
   - 线性基准：OLS (作为线性概率模型), Logit (Ridge), LASSO
   - 支持向量机：Linear SVM (使用 CalibratedClassifierCV 获取概率, cv=3加速)
   - 树模型/集成学习：Decision Tree (深度10), Random Forest (深度15), LightGBM (Optuna优化参数)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, cohen_kappa_score,
    r2_score
)

# ==============================================================================
# 0. 全局配置区
# ==============================================================================
TRAIN_FILE = 'ready_train.parquet'
TEST_FILE = 'ready_test.parquet'

FEATURE_COLS = [
    'Accum_Vol_Diff', 
    'VolumeMax', 
    'VolumeAll', 
    'Immediacy', 
    'Depth_Change', 
    'LobImbalance', 
    'DeepLobImbalance', 
    'Relative_Spread', 
    'Micro_Mid_Spread', 
    'PastReturn', 
    'Lambda', 
    'Volatility', 
    'AutoCov'
]

MODELS_TO_RUN = [
   'OLS', 'Logit', 'LASSO', 'SVM', 
    'Decision Tree', 'Random Forest', 'LightGBM'
] 

# [新增] 确保模型保存目录存在
if not os.path.exists('models'):
    os.makedirs('models')

# ==============================================================================
# 1. 数据处理与辅助函数
# ==============================================================================

def load_ready_data():
    """
    功能：加载预处理好的 Parquet 数据集。
    输入：全局配置的 TRAIN_FILE 和 TEST_FILE 路径。
    输出：
        - X_train, X_test: 特征矩阵 (numpy array)
        - y_train, y_test: 标签向量 (numpy array)
    逻辑：
        1. 读取 Parquet 文件。
        2. 自动识别标签列 'Label' 和特征列。
        3. 转换为 Numpy 格式以适配 Scikit-learn 接口。
    """
    print(f"正在读取预处理文件: {TRAIN_FILE} 和 {TEST_FILE} ...")
    try:
        train_df = pd.read_parquet(TRAIN_FILE)
        test_df = pd.read_parquet(TEST_FILE)
    except Exception as e:
        print(f"错误: 读取文件失败。请检查文件名或是否已运行预处理脚本。\n{e}")
        exit()

    print(f"加载完成! 训练集: {train_df.shape}, 测试集: {test_df.shape}")

    label_col = 'Label'

    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    
    if len(available_features) != len(FEATURE_COLS):
        print(f"警告: 部分定义在 FEATURE_COLS 中的特征未在数据中找到: {set(FEATURE_COLS) - set(available_features)}")
    
    X_train = train_df[available_features].values
    y_train = train_df[label_col].values
    X_test = test_df[available_features].values
    y_test = test_df[label_col].values

    return X_train, X_test, y_train, y_test

def _package_results(model, X_train, X_test):
    """
    功能：统一封装模型的预测结果。
    输入：模型对象, 训练集特征, 测试集特征。
    输出：字典，包含模型对象、训练/测试预测标签、测试集概率。
    逻辑：
        1. 获取 predict() 预测的类别标签。
        2. 若模型支持 predict_proba()，获取正类概率；否则设为 None。
    """
    results = {
        'model': model,
        'train_pred': model.predict(X_train),
        'test_pred': model.predict(X_test),
    }
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        # 二分类情况取第1列作为正类概率
        if probs.shape[1] == 2:
            results['test_proba'] = probs[:, 1]
        else:
            results['test_proba'] = None 
    else:
        results['test_proba'] = None
    return results

# ==============================================================================
# 2. 模型训练函数
# ==============================================================================

def train_ols(X_train, y_train, X_test):
    """
    功能：训练 OLS 线性回归模型 (线性概率模型 LPM)。
    逻辑：
        1. 拟合 LinearRegression。
        2. 将连续预测值按 0.5 阈值二值化为分类标签。
        3. 保留原始连续值作为 'test_proba' 用于 AUC/AP 计算。
    """
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    
    raw_train_pred = model.predict(X_train)
    raw_test_pred = model.predict(X_test)
    
    # 阈值分类 (>=0.5 为 1, 否则为 0)
    train_pred_cls = np.where(raw_train_pred >= 0.5, 1, 0)
    test_pred_cls = np.where(raw_test_pred >= 0.5, 1, 0)
    
    results = {
        'model': model,
        'train_pred': train_pred_cls,
        'test_pred': test_pred_cls,
        'test_proba': raw_test_pred 
    }
    return results

def train_logit(X_train, y_train, X_test):
    """功能：训练 Logistic 回归 (Ridge)。逻辑：L2正则, n_jobs=-1 并行。"""
    model = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, 
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return _package_results(model, X_train, X_test)

def train_lasso(X_train, y_train, X_test):
    """功能：训练 Logistic 回归 (Lasso)。逻辑：L1正则, solver='saga' 适合大数据。"""
    model = LogisticRegression(
        penalty='l1', C=0.1, solver='saga', max_iter=1000, 
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return _package_results(model, X_train, X_test)

def train_svm(X_train, y_train, X_test):
    """功能：训练线性 SVM。逻辑：使用 LinearSVC 加速，CalibratedClassifierCV(cv=3) 获取概率。"""
    base_model = LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, dual=False, random_state=42)
    model = CalibratedClassifierCV(base_model, cv=3, n_jobs=-1) 
    model.fit(X_train, y_train)
    return _package_results(model, X_train, X_test)

def train_tree(X_train, y_train, X_test):
    """功能：训练决策树。逻辑：最大深度10，防止在大数据下欠拟合。"""
    model = DecisionTreeClassifier(
        max_depth=10, class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)
    return _package_results(model, X_train, X_test)

def train_rf(X_train, y_train, X_test):
    """功能：训练随机森林。逻辑：100棵树, 最大深度15, 平衡性能与速度。"""
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, class_weight='balanced', 
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return _package_results(model, X_train, X_test)

def train_lgbm(X_train, y_train, X_test):
    """
    功能：训练 LightGBM。
    逻辑：使用 Optuna 贝叶斯优化得到的最佳参数组合 (AUC ~0.78)。
         核心参数：num_leaves=121, learning_rate=0.05, 树数量=500。
         引入了随机采样 (bagging/feature_fraction) 和正则化以抗过拟合。
    """
    model = lgb.LGBMClassifier(
        n_estimators=500,                    
        learning_rate=0.05002610359120165,
        num_leaves=121,                      
        max_depth=15,
        
        # 正则化与采样参数
        reg_alpha=0.002323904305642182,      # lambda_l1
        reg_lambda=1.5691440113196792e-07,   # lambda_l2
        colsample_bytree=0.6788114724629921, # feature_fraction
        subsample=0.6132806192183446,        # bagging_fraction
        subsample_freq=6,                    # bagging_freq
        min_child_samples=67,
        
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return _package_results(model, X_train, X_test)

MODEL_DISPATCHER = {
    'OLS': train_ols, 'Logit': train_logit, 'LASSO': train_lasso, 
    'SVM': train_svm, 'Decision Tree': train_tree, 
    'Random Forest': train_rf, 'LightGBM': train_lgbm 
}

# ==============================================================================
# 3. 可视化与评估函数
# ==============================================================================

def compare_models_visuals(results_store, y_test, feature_names=None):
    """
    功能：计算各模型评估指标并绘制四组对比图表。
    输入：
        - results_store: 包含各模型预测结果的字典
        - y_test: 真实标签
        - feature_names: 特征名称列表
    输出：
        - 控制台打印评估指标表。
        - 保存混淆矩阵、特征重要性、ROC/PR曲线图片。
    逻辑：
        1. 遍历模型计算 Accuracy, F1, AUC, LogLoss 等指标。
           * 特别注意：对 OLS 的 LogLoss 计算进行 try-except 处理。
        2. 绘制混淆矩阵 (SVM 单独绘制，其他模型 2x3 网格)。
        3. 提取特征重要性，进行 MaxAbs 归一化以统一量纲。
        4. 绘制 ROC 和 PR 曲线。
    """
    model_names = list(results_store.keys())
    n_models = len(model_names)
    if n_models == 0: return

    # 1. 自动识别类别信息
    classes = np.unique(y_test)
    n_classes = len(classes)
    is_multiclass = n_classes > 2
    avg_method = 'binary' if not is_multiclass else 'weighted'
    
    cmap_base = plt.get_cmap('tab10')
    colors = [cmap_base(i) for i in range(n_models)]
    color_map = {name: colors[i] for i, name in enumerate(model_names)}
    
    # -----------------------------------------------------------
    # 1. 计算与打印指标表
    # -----------------------------------------------------------
    metrics_list = []
    y_test_bin = label_binarize(y_test, classes=classes) if is_multiclass else None

    for name in model_names:
        y_pred = results_store[name]['test_pred']
        y_proba = results_store[name]['test_proba']
        
        # 基础指标
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method)
        f1 = f1_score(y_test, y_pred, average=avg_method)
        
        # 初始化概率指标
        auc_val, ap, ll, r2 = np.nan, np.nan, np.nan, np.nan
        
        if y_proba is not None:
            r2 = r2_score(y_test, y_proba)
            try:
                # 尝试直接计算
                ll = log_loss(y_test, y_proba)
            except ValueError:
                # 如果报错（通常是因为OLS输出了负数或大于1的数），则进行截断处理
                # 将数值限制在 [epsilon, 1-epsilon] 之间
                eps = 1e-15
                y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
                ll = log_loss(y_test, y_proba_clipped)

            # -------------------------------------------------
            # AUC 和 AP 计算
            # -------------------------------------------------
            try:
                if is_multiclass:
                    auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    ap = average_precision_score(y_test_bin, y_proba, average='weighted')
                else:
                    auc_val = roc_auc_score(y_test, y_proba)
                    ap = average_precision_score(y_test, y_proba)
            except Exception as e:
                print(f"Warn: 模型 {name} 计算 AUC/AP 失败: {e}")

        metrics_list.append({
            'Model': name, 'F1-Score': f1, 'AUC': auc_val, 'AP': ap, 
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
            'Kappa': kappa, 'LogLoss': ll, 'R-Squared': r2
        })
        
    df_metrics = pd.DataFrame(metrics_list).set_index('Model')
    df_metrics = df_metrics.sort_values(by='F1-Score', ascending=False)
    
    print("\n" + "="*60)
    print(f" 模型评估指标汇总 (按 F1-Score 排序, Average={avg_method})")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_metrics.round(4))
    print("="*60 + "\n")

    # -----------------------------------------------------------
    # 2. 绘制混淆矩阵
    # -----------------------------------------------------------
    
    # A. SVM (单独绘制)
    if 'SVM' in results_store:
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(y_test, results_store['SVM']['test_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(color_map['SVM'], as_cmap=True), 
                    square=True, cbar=False, annot_kws={"size": 20, "weight": "bold"},
                    xticklabels=['B', 'S'], yticklabels=['B', 'S'])
        plt.title("SVM", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=14); plt.ylabel('True', fontsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix_svm.png', dpi=300)
        plt.close()

    # B. 其他模型 (2x3 布局)
    others_layout = [['OLS', 'Logit', 'LASSO'], ['Decision Tree', 'Random Forest', 'LightGBM']]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            name = others_layout[r][c]
            if name in results_store:
                cm = confusion_matrix(y_test, results_store[name]['test_pred'])
                col = color_map.get(name, 'gray')
                sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(col, as_cmap=True), 
                            ax=ax, square=True, cbar=False, annot_kws={"size": 20, "weight": "bold"},
                            xticklabels=['B', 'S'], yticklabels=['B', 'S'])
                ax.set_title(name, color=col, fontweight='bold', fontsize=16, pad=15)
                ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
            else:
                ax.axis('off')
    plt.tight_layout()
    plt.savefig('confusion_matrix_others.png', dpi=300)
    plt.close()

    # -----------------------------------------------------------
    # 3. 绘制特征重要性 (归一化)
    # -----------------------------------------------------------
    if feature_names is not None:
        importance_data = {}
        for name in model_names:
            model = results_store[name]['model']
            imps = None
            
            # 提取重要性或系数
            if hasattr(model, 'feature_importances_'):
                imps = model.feature_importances_
            elif hasattr(model, 'coef_'):
                imps = model.coef_
            elif hasattr(model, 'calibrated_classifiers_'):
                 coefs = [clf.estimator.coef_ if hasattr(clf, 'estimator') else clf.base_estimator.coef_ 
                          for clf in model.calibrated_classifiers_]
                 if coefs: imps = np.mean(coefs, axis=0)

            # 归一化处理 (除以最大绝对值)
            if imps is not None:
                imps = np.array(imps).flatten() 
                if len(imps) == len(feature_names):
                    max_val = np.max(np.abs(imps))
                    if max_val > 0: imps = imps / max_val
                    importance_data[name] = imps
        
        if importance_data:
            df_imp = pd.DataFrame(importance_data, index=feature_names)
            plt.figure(figsize=(14, max(8, len(feature_names) * 0.5)))
            # RdBu_r: 蓝负红正
            sns.heatmap(df_imp, annot=True, fmt=".2f", cmap="RdBu_r", center=0, 
                        cbar_kws={'label': 'Normalized Importance (Scaled by MaxAbs)'}, annot_kws={"size": 10})
            plt.title('Normalized Feature Importance Comparison', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig('feature_importance_normalized.png', dpi=300)
            plt.close()

    # -----------------------------------------------------------
    # 4. 绘制 ROC / PR 曲线
    # -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for name in model_names:
        y_proba = results_store[name]['test_proba']
        col = color_map[name]
        if y_proba is None: continue
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax1.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
        
        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax2.plot(recall, precision, color=col, lw=2, label=f'{name} (AP={average_precision_score(y_test, y_proba):.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5); ax1.set_title('ROC Curve'); ax1.legend(loc="lower right")
    ax2.set_title('Precision-Recall Curve'); ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=300)
    plt.close()

# ==============================================================================
# 4. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 1. 准备数据
    X_train, X_test, y_train, y_test = load_ready_data()
    
    # 2. 批量训练
    results_store = {}
    print("\n" + "="*50)
    print(f" 开始批量训练 {len(MODELS_TO_RUN)} 个模型...")
    print("="*50)

    for i, name in enumerate(MODELS_TO_RUN, 1):
        if name in MODEL_DISPATCHER:
            current_time = time.strftime('%H:%M:%S')
            print(f"[{i}/{len(MODELS_TO_RUN)}] {current_time} | 正在训练 {name} ... ", end='', flush=True)
            
            start_time = time.time()
            
            # 执行训练
            results_store[name] = MODEL_DISPATCHER[name](X_train, y_train, X_test)
            
            # 保存模型
            model_to_save = results_store[name]['model']
            save_path = f"models/{name.replace(' ', '_')}.pkl"
            joblib.dump(model_to_save, save_path)

            elapsed = time.time() - start_time
            print(f"完成! (耗时: {elapsed:.2f}秒) | 模型已保存至: {save_path}")
        else:
            print(f"\n警告: 模型 {name} 未定义。")

    print("="*50)
    print(" 所有模型训练结束，正在生成报告...")
    print("="*50)

    # 3. 生成分析报告与图表
    compare_models_visuals(
        results_store=results_store, 
        y_test=y_test, 
        feature_names=FEATURE_COLS
    )