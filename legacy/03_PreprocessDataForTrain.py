"""
程序名称：高频数据预处理与标签生成 (Data Preprocessing & Labeling)

1. 程序概述：
   本程序负责将原始合并后的高频行情数据转换为机器学习模型可直接读取的训练/测试集。
   核心任务包括：构建预测标签（未来价格变动方向）、清洗无效数据（去极值/空值）、
   严格按时间轴划分数据集、以及特征标准化处理。这是连接原始数据与模型训练的关键桥梁。

2. 输入数据：
   - 输入文件：'market_data.parquet' (由 02_ConcatDataFromDB.py 生成的沪深合并数据)
   - 特征列(X)：13个核心微观结构特征，包括：
     Accum_Vol_Diff, VolumeMax, VolumeAll, Immediacy, Depth_Change,
     LobImbalance, DeepLobImbalance, Relative_Spread, Micro_Mid_Spread,
     PastReturn, Lambda, Volatility, AutoCov

3. 核心逻辑流程：
   1) 标签生成 (Label Generation)：
      - 目标：预测下一个时间步（3秒后）的中间价变动方向。
      - 逻辑：计算 Next_MidPrice - MidPrice。
      - 过滤：剔除价格无变化 (Diff=0) 的样本，构建二分类任务 (涨=1, 跌=0)。
   
   2) 强力清洗 (Deep Cleaning)：
      - 异常值处理：将无穷大 (inf/-inf) 替换为 NaN。
      - 空值剔除：彻底删除任何包含 NaN 的行。
        * 特别注意：AutoCov 和 Volatility 等特征依赖历史窗口，每天开盘前几秒通常为 NaN，
          必须在此步骤剔除，否则会导致后续线性模型 (OLS/Lasso) 报错。

   3) 数据集划分 (Time-based Split)：
      - 原则：严格按照时间顺序切分，严禁 Shuffle。
      - 方法：提取所有唯一时间戳并排序，以 80% 分位点为界。
      - 结果：训练集包含所有股票的“早期”数据，测试集包含所有股票的“晚期”数据。

   4) 特征标准化 (Normalization)：
      - 方法：使用 StandardScaler 进行 Z-Score 标准化 (均值0，方差1)。
      - 防泄露：仅使用【训练集】拟合 Scaler，然后将其应用于测试集。

4. 输出内容：
   - 训练集文件：'ready_train.parquet' (特征已标准化，包含 Label 列)
   - 测试集文件：'ready_test.parquet'
   - 控制台日志：清洗行数统计、标签分布、切分时间点、最终样本量等。
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================
INPUT_FILE = 'market_data.parquet'
OUTPUT_TRAIN_FILE = 'ready_train.parquet'
OUTPUT_TEST_FILE = 'ready_test.parquet'

# 新版特征列表 (13个核心特征)
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

def preprocess_and_save():
    """
    主处理流程：读取 -> 清洗 -> 标注 -> 切分 -> 标准化 -> 导出
    """
    start_global = time.time()
    
    # ==========================================================================
    # 1. 读取数据 (Data Loading)
    # ==========================================================================
    print(f">>> [1/6] 正在读取原始文件: {INPUT_FILE} ...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"读取失败: {e}")
        return
    
    print(f"    原始数据形状: {df.shape}")

    # ==========================================================================
    # 2. 标签生成 (Label Generation)
    # ==========================================================================
    print(f">>> [2/6] 正在生成预测标签 ...")
    
    # 逻辑：使用 shift(-1) 获取下一时刻(3秒后)的中间价
    # 注意：必须按 Stock 分组 shift，否则不同股票间的数据会错位
    df['Next_MidPrice'] = df.groupby('Stock')['MidPrice'].shift(-1)
    
    # 计算价格变动 diff
    df['Target_Diff'] = df['Next_MidPrice'] - df['MidPrice']
    
    # 移除最后一行 (因为没有未来数据，diff 为 NaN)
    df.dropna(subset=['Target_Diff'], inplace=True)
    
    # 过滤平盘数据 (Diff=0)，将问题转化为二分类 (涨 vs 跌)
    # 平盘通常噪音大且难以预测，剔除有助于模型聚焦显著变化
    df_clean = df[df['Target_Diff'] != 0].copy()
    
    # 生成 Label: 涨(>0) -> 1, 跌(<0) -> 0
    df_clean['Label'] = (df_clean['Target_Diff'] > 0).astype(int)
    
    print(f"    标签生成完毕。当前样本数: {len(df_clean)}")

    # ==========================================================================
    # 3. 数据清洗 (Deep Cleaning)
    # ==========================================================================
    print(f">>> [3/6] 正在进行强力清洗 (去除特征 NaN/Inf) ...")
    
    # 1. 处理无穷大值 (通常由除以0产生)
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. 剔除空值行
    # 注意：AutoCov, Volatility 等依赖历史窗口的特征在每天开盘前几秒通常为 NaN
    # 必须将其剔除，否则会导致后续线性模型 (OLS/Lasso) 无法训练
    before_drop = len(df_clean)
    df_clean.dropna(subset=FEATURE_COLS, inplace=True)
    after_drop = len(df_clean)
    
    print(f"    清洗掉的脏数据行数: {before_drop - after_drop}")
    print(f"    清洗后剩余样本数: {after_drop}")
    print(f"    标签分布: \n{df_clean['Label'].value_counts(normalize=True)}")

    # ==========================================================================
    # 4. 数据集划分 (Time-Series Split with Day Alignment)
    # ==========================================================================
    print(f">>> [4/6] 正在划分数据集 (前80%训练, 后20%测试) ...")
    
    # 1. 全局排序：确保时间轴单调递增，防止未来信息泄露
    df_clean.sort_values(by=['Stock', 'Time'], inplace=True)
    
    # 2. 提取唯一时间戳序列
    unique_times = df_clean['Time'].unique()
    unique_times = np.sort(unique_times) 
    
    # 3. 计算 80% 分位点
    split_idx_raw = int(len(unique_times) * 0.8)
    split_time_raw = unique_times[split_idx_raw]
    
    print(f"    原始切分时间点: {split_time_raw}")
    
    # [核心逻辑] 边界智能对齐
    # 如果切分点恰好落在某天收盘 (如 15:00:00)，直接切分会导致测试集的第一条数据
    # 是次日开盘数据，但其“历史回看窗口”可能会错误地连接到前一日收盘。
    # 策略：如果检测到切分点位于日末，将切分点后移至次日开盘，确保测试集从完整的新一天开始。
    if split_idx_raw + 1 < len(unique_times):
        next_time = unique_times[split_idx_raw + 1]
        
        t1 = pd.to_datetime(split_time_raw)
        t2 = pd.to_datetime(next_time)
        
        # 如果相邻两个时间点跨越了日期 (即跨日)
        if t1.date() != t2.date():
            print(f"    [Info] 检测到原始切分点 {t1} 位于日末。")
            print(f"    [Action] 微调切分点至 {t2}，确保测试集从新的一天开始。")
            split_time_final = next_time
        else:
            # 未跨日，保持原切分点
            split_time_final = unique_times[split_idx_raw + 1]
    else:
        split_time_final = split_time_raw

    print(f"    最终切分时间点: {split_time_final}")
    
    # 执行切分
    train_mask = df_clean['Time'] < split_time_final
    test_mask = df_clean['Time'] >= split_time_final
    
    df_train = df_clean[train_mask].copy()
    df_test = df_clean[test_mask].copy()
    
    print(f"    训练集数量: {len(df_train)}")
    print(f"    测试集数量: {len(df_test)}")

    # ==========================================================================
    # 5. 特征标准化 (Standardization)
    # ==========================================================================
    print(f">>> [5/6] 正在进行特征归一化 (StandardScaler) ...")
    
    # 准备 numpy 数组
    X_train_raw = df_train[FEATURE_COLS].values
    y_train = df_train['Label'].values
    
    X_test_raw = df_test[FEATURE_COLS].values
    y_test = df_test['Label'].values

    # 初始化并拟合 Scaler
    # [关键] 仅使用训练集拟合 (fit)，以避免“未来数据泄露”
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # 将训练集的统计量应用到测试集
    X_test_scaled = scaler.transform(X_test_raw)

    # ==========================================================================
    # 6. 格式化与导出 (Formatting & Export)
    # ==========================================================================
    print(f">>> [6/6] 正在保存结果至 Parquet ...")
    
    # 重组 DataFrame
    train_export = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    train_export['Label'] = y_train
    
    test_export = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS)
    test_export['Label'] = y_test

    # [修复 1] 补全元数据列 (Stock, Time)
    # 必须补全这些列，后续序列模型 (RNN/GRU) 需要依据 Stock 分组和 Time 排序
    # 同时修复 Stock 代码被转为数字的问题 (如 1 -> '000001')
    train_export['Stock'] = df_train['Stock'].astype(str).str.zfill(6).values
    train_export['Time'] = df_train['Time'].values
    
    test_export['Stock'] = df_test['Stock'].astype(str).str.zfill(6).values
    test_export['Time'] = df_test['Time'].values

    # 调整列顺序，方便阅读
    cols_order = ['Stock', 'Time'] + FEATURE_COLS + ['Label']
    train_export = train_export[cols_order]
    test_export = test_export[cols_order]

    # [修复 2] 最终强制排序
    # 保证数据物理存储顺序为：先按股票聚类，再按时间递增。
    # 这对于 sliding window (滑动窗口) 操作至关重要。
    print("    正在对训练集进行最终排序 (Stock -> Time)...")
    train_export.sort_values(by=['Stock', 'Time'], inplace=True)
    
    print("    正在对测试集进行最终排序 (Stock -> Time)...")
    test_export.sort_values(by=['Stock', 'Time'], inplace=True)
    
    # 打印最终时间范围确认
    if not train_export.empty:
        print(f"    Train Time Range: {train_export['Time'].min()} ~ {train_export['Time'].max()}")
    if not test_export.empty:
        print(f"    Test  Time Range: {test_export['Time'].min()} ~ {test_export['Time'].max()}")

    # 写入文件 (使用 snappy 压缩平衡速度与体积)
    train_export.to_parquet(OUTPUT_TRAIN_FILE, compression='snappy')
    test_export.to_parquet(OUTPUT_TEST_FILE, compression='snappy')

    print(f"\n====== 处理完成! ======")
    print(f"总耗时: {time.time() - start_global:.2f} 秒")
    print(f"训练集已保存: {OUTPUT_TRAIN_FILE}")
    print(f"测试集已保存: {OUTPUT_TEST_FILE}")

if __name__ == "__main__":
    preprocess_and_save()