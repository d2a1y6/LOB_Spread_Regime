"""
程序名称：基于 Optuna 的 LightGBM 自动化调参工具 (Hyperparameter Tuning Tool)

1. 程序概述：
   本程序专为海量高频交易数据设计，利用 Optuna 框架（贝叶斯优化 TPE 算法）对 LightGBM 模型进行自动化超参数搜索。
   旨在寻找能够最大化验证集 AUC 的最优参数组合，涵盖学习率、树结构、正则化及随机采样等关键参数。
   程序包含数据加载、时序验证集切分、早停机制 (Early Stopping)、全量重训练及最终评估的全流程。

2. 输入数据：
   - 训练集文件：'ready_train.parquet' (用于划分 训练子集 和 验证子集)
   - 测试集文件：'ready_test.parquet' (仅用于最终模型的独立评估)
   - 特征列(X)：自动识别 Parquet 文件中除 'Label' 外的所有列。
   - 标签列(y)：'Label' (二分类标签)。

3. 核心方法：
   - 优化算法：Optuna (TPE - Tree-structured Parzen Estimator)
   - 基模型：LightGBM Classifier
   - 评估指标：ROC-AUC (Area Under Curve)

4. 核心逻辑流程：
   1) 数据加载：读取预处理好的 Parquet 文件。
   2) 验证集划分：在训练集内部，严格按照时间顺序切分最后 20% 作为验证集 (Validation Set)，用于 Optuna 的早停和评分。
   3) 参数搜索：
      - 定义搜索空间：learning_rate, num_leaves, max_depth, lambda_l1/l2, bagging/feature_fraction 等。
      - 执行优化：运行 N_TRIALS 次试验，每次试验使用早停机制 (Early Stopping) 防止过拟合。
   4) 最佳模型重训练：
      - 获取最佳参数组合。
      - 增加树的数量 (n_estimators=500)。
      - 使用【全量训练集】(Train + Val) 重新训练最终模型。
   5) 最终评估与保存：
      - 在独立的测试集上计算 AUC、Accuracy 及生成分类报告。
      - 保存训练好的模型文件 (.pkl)。
      - 输出特征重要性排名。

5. 输出内容与位置：
   - 控制台(Console)：打印优化过程、最佳参数、最终测试集评估指标、Top 5 特征重要性。
   - 模型文件(当前目录)：'best_lgbm_model.pkl' (可用于后续加载和预测)。
"""

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import time
import os
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
TRAIN_FILE = 'ready_train.parquet'
TEST_FILE = 'ready_test.parquet'
MODEL_SAVE_PATH = 'best_lgbm_model.pkl'

# 设定优化的试验次数 (建议 50-100 次以获得较好结果)
N_TRIALS = 30 

def load_data():
    """
    功能：加载 Parquet 数据并分离特征与标签。
    输出：训练集与测试集的特征矩阵(X)和标签向量(y)，以及特征名称列表。
    """
    print(f">>> 正在加载数据...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)
    
    label_col = 'Label'
    feature_cols = [c for c in train_df.columns if c != label_col]
    
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[label_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values
    
    # 释放内存
    del train_df, test_df
    
    print(f"    训练集: {X_train_full.shape}")
    print(f"    测试集: {X_test.shape}")
    
    return X_train_full, y_train_full, X_test, y_test, feature_cols

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna 的目标函数：定义参数搜索空间并返回验证集分数。
    逻辑：
        1. 从搜索空间采样参数。
        2. 构建 LGBMClassifier。
        3. 在 (X_train, y_train) 上训练，在 (X_val, y_val) 上验证并执行早停。
        4. 返回验证集的 AUC。
    """
    # 定义参数搜索空间
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'class_weight': 'balanced',
        
        # 核心结构参数
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'n_estimators': 300, # 配合 early_stopping 使用，仅用于搜索阶段
        
        # 正则化与随机采样 (抗过拟合)
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    # 训练模型 (使用早停机制)
    model = lgb.LGBMClassifier(**param)
    
    # 定义早停回调
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0) # 不打印过程日志
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=callbacks
    )
    
    # 返回验证集的 AUC 作为优化目标
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

def run_optimization():
    # 1. 加载数据
    X_train_full, y_train_full, X_test, y_test, features = load_data()
    
    # 2. 划分验证集 (从训练集中切分出最后 20% 用于调参验证)
    #    注意：shuffle=False 严格保持时序性 (按股票/时间排序的数据)
    print(f">>> 正在划分调参用的验证集 (20% of Train)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=False
    )
    
    # 3. 创建 Optuna 研究对象并执行优化
    print(f">>> 开始 Optuna 自动化调参 ({N_TRIALS} 次试验)...")
    study = optuna.create_study(direction='maximize')
    
    # 使用 lambda 包装目标函数以传递数据参数
    func = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    
    study.optimize(func, n_trials=N_TRIALS)

    print("\n" + "="*50)
    print(" 调参完成！")
    print("="*50)
    print(f"最佳 AUC: {study.best_value:.4f}")
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # 4. 使用最佳参数重新训练最终模型
    print(f"\n>>> 使用最佳参数在全量训练集上训练最终模型...")
    
    best_params = study.best_params
    # 补充固定参数与增强配置
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_estimators': 500  # 最终模型增加树的数量以提升容量
    })
    
    final_model = lgb.LGBMClassifier(**best_params)
    
    start_time = time.time()
    final_model.fit(X_train_full, y_train_full)
    print(f"训练耗时: {time.time() - start_time:.2f} 秒")
    
    # 5. 最终评估
    print(f"\n>>> 最终测试集评估:")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    print(f"Test AUC:      {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. 保存模型与特征重要性
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {MODEL_SAVE_PATH}")
    
    print("\nTop 5 特征重要性:")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': final_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(importance.head(5))

if __name__ == '__main__':
    run_optimization()