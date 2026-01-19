"""
程序名称：基于 Attention-GRU 的时序注意力机制模型 (GPU 修复版)

1. 程序概述：
   本程序实现了一个带有注意力机制（Attention Mechanism）的 GRU 深度学习模型。
   旨在验证在预测未来价格方向时，历史窗口中是否某些特定时刻（如大单成交瞬间）具有更高的权重。

   [修复说明]：
   - 修复了数据加载时包含非数值列 (如 Timestamp) 导致 StandardScaler 报错的问题。
   - 明确指定了 13 维微观结构特征，过滤掉时间戳等无关信息。

2. 输入数据：
   - 训练集：'ready_train.parquet'
   - 测试集：'ready_test.parquet'
   - 数据形态：(Batch_Size, Seq_Len, Input_Dim)
     * Seq_Len: 默认为 20 (即过去 60秒的数据)。
     * Input_Dim: 13维微观结构特征。

3. 模型结构 (GRUWithAttention)：
   - 编码层：GRU (双层) -> 捕捉时序依赖，返回所有时间步的 Hidden States。
   - 注意力层：Bahdanau Attention -> 计算时间步权重，生成 Context Vector。
   - 输出层：全连接层 -> 输出 Logits (配合 BCEWithLogitsLoss)。

4. 功能特性：
   - 平台自适应：自动识别 Windows(CUDA 4090) 环境，启用高性能 DataLoader 配置。
   - 性能优化：使用 persistent_workers 和 pin_memory 消除数据加载瓶颈。
   - 静态绘图：训练结束后自动保存 Loss/AUC/Acc 曲线图至 models/ 目录。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，只保存图片不弹窗
import matplotlib.pyplot as plt
import os
import gc
import time

# ==============================================================================
# 0. 全局配置区 (平台自适应)
# ==============================================================================
TRAIN_FILE = 'ready_train.parquet'
TEST_FILE = 'ready_test.parquet'

# [关键参数] 修改此处 SEQ_LEN 可进行“市场记忆性”实验
SEQ_LEN = 20           # 20个 snapshot * 3s = 60s 历史窗口
LEARNING_RATE = 1e-3
EPOCHS = 100            
PATIENCE = 15
DROPOUT_RATE = 0.3     
WEIGHT_DECAY = 1e-4    

# 检测运行环境并配置硬件参数
if torch.cuda.is_available():
    # --- Windows / Linux with NVIDIA GPU (如 RTX 4090) ---
    DEVICE = torch.device('cuda')
    # GRU 计算量比 MLP 大，但 4090 显存充足，使用大 Batch 压榨算力
    BATCH_SIZE = 16384      
    NUM_WORKERS = 4         # CPU 多线程加载
    PIN_MEMORY = True       # 锁页内存，加速 CPU->GPU 传输
    PERSISTENT_WORKERS = True # 保持 worker 进程存活
    # print(f">> 检测到 CUDA 环境 ({torch.cuda.get_device_name(0)})，已启用高性能配置 (Batch={BATCH_SIZE})。")
else:
    # --- Mac (MPS) / CPU ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print(">> 检测到 Mac MPS 环境，使用标准配置。")
    else:
        DEVICE = torch.device('cpu')
        print(">> 检测到 CPU 环境，使用标准配置。")
    
    BATCH_SIZE = 2048       # Mac/CPU 适当减小 Batch Size
    NUM_WORKERS = 0         # Mac 多进程有时不稳定
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False

# 确保模型保存目录存在
if not os.path.exists('models'):
    os.makedirs('models')

# ==============================================================================
# 1. 工具类定义
# ==============================================================================

class TrainingVisualizer:
    """
    功能：静态绘图工具，用于在训练结束后生成并保存评估曲线。
    逻辑：记录每个 Epoch 的指标，save_plot() 方法绘制双轴图表并保存。
    """
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.val_aucs = []
        self.val_accs = []
        
    def update(self, epoch, loss, val_auc, val_acc):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.val_aucs.append(val_auc)
        self.val_accs.append(val_acc)
        
    def save_plot(self, filename):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制左轴 (Loss)
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.epochs, self.losses, color=color, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        # 绘制右轴 (Metrics)
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('AUC / Accuracy', color=color) 
        ax2.plot(self.epochs, self.val_aucs, color=color, marker='s', linestyle='-', label='Val AUC')
        ax2.plot(self.epochs, self.val_accs, color='green', marker='^', linestyle='--', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Attention-GRU Training Metrics (Seq_Len={SEQ_LEN})')
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        plt.close()
        print(f"训练曲线已保存至: {filename}")

class TimeSeriesDataset(Dataset):
    """
    功能：构建适用于 RNN/GRU 的时序数据集。
    输入：
        - data: 特征矩阵 (N, Feature_Dim)
        - labels: 标签向量 (N,)
        - seq_len: 时间窗口长度
    处理逻辑：
        - __getitem__ 动态切片返回 (seq_len, feature_dim) 作为输入 X。
    """
    def __init__(self, data, labels, seq_len):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 取 [idx, idx+seq_len) 作为序列特征
        # 取 idx+seq_len-1 作为对应的标签时间点
        return self.data[idx : idx + self.seq_len], self.labels[idx + self.seq_len - 1]

# ==============================================================================
# 2. 模型结构定义 (Attention-GRU)
# ==============================================================================

class Attention(nn.Module):
    """
    功能：实现 Bahdanau 风格的注意力机制。
    
    输入：GRU 所有时间步的输出 (Batch, Seq_Len, Hidden_Dim)。
    输出：加权后的 Context Vector 和 注意力权重分布。
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, gru_outputs):
        # 1. 计算能量分数: (Batch, Seq_Len, 1)
        energy = self.attention_layer(gru_outputs) 
        
        # 2. 计算权重 (Alpha): (Batch, Seq_Len, 1)
        weights = F.softmax(energy, dim=1)         
        
        # 3. 加权求和得到 Context Vector: (Batch, Hidden_Dim)
        context_vector = torch.sum(weights * gru_outputs, dim=1)
        
        return context_vector, weights

class GRUWithAttention(nn.Module):
    """
    模型架构：
    Input -> GRU (Layer=2) -> Attention Layer -> Fully Connected -> Logits
    
    优化：移除最终的 Sigmoid 层，配合 BCEWithLogitsLoss 使用以提升数值稳定性。
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUWithAttention, self).__init__()
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT_RATE if num_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = Attention(hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(32, 1) # 移除 Sigmoid
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        
        # GRU 输出: (Batch, Seq_Len, Hidden_Dim)
        gru_out, _ = self.gru(x)
        
        # Attention 融合: (Batch, Hidden_Dim)
        context, _ = self.attention(gru_out)
        
        # 分类输出 (Logits)
        output = self.fc(context)
        return output

# ==============================================================================
# 3. 训练与评估流程
# ==============================================================================

def run_pipeline():
    """
    主流程函数：执行数据加载、模型训练、早停保存及最终测试。
    """
    print(f"启动 Attention-GRU 训练流程 (Seq_Len: {SEQ_LEN})...")
    start_time = time.time()
    
    model_filename = f'models/best_attn_gru_seq{SEQ_LEN}.pth'
    
    # --- 1. 数据加载与修复 ---
    print("正在加载 Parquet 数据...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    # [修复点] 显式定义数值型特征，避免读取 Timestamp 等非数值列导致报错
    target_features = [
        'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
        'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
        'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
    ]
    # 取交集确保列存在
    feature_cols = [c for c in target_features if c in train_df.columns]
    
    print(f"使用特征 ({len(feature_cols)}维): {feature_cols}")

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df['Label'].values
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df['Label'].values

    # 标准化
    scaler = StandardScaler()
    # 此时 X_train_raw 仅包含数值，不会再报 Timestamp 错误
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 释放内存
    del train_df, test_df, X_train_raw, X_test_raw
    gc.collect()

    # --- 2. 构建 Dataset 与 DataLoader ---
    split_idx = int(len(X_train_scaled) * 0.8)
    
    train_dataset = TimeSeriesDataset(X_train_scaled[:split_idx], y_train_raw[:split_idx], SEQ_LEN)
    val_dataset = TimeSeriesDataset(X_train_scaled[split_idx:], y_train_raw[split_idx:], SEQ_LEN)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_raw, SEQ_LEN)

    # 针对 Windows/CUDA 优化的 DataLoader 参数
    loader_args = {
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY
    }
    if PERSISTENT_WORKERS and NUM_WORKERS > 0:
        loader_args['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    print(f"数据加载完成: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"当前配置: Device={DEVICE}, BatchSize={BATCH_SIZE}, Workers={NUM_WORKERS}")

    # --- 3. 初始化模型 ---
    model = GRUWithAttention(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 优化：使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    visualizer = TrainingVisualizer()
    best_auc = 0
    patience_counter = 0

    # --- 4. 训练循环 ---
    print("开始训练...")
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_preds_prob, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(DEVICE)
                outputs = model(X_b)
                # 手动加 Sigmoid 获取概率
                probs = torch.sigmoid(outputs)
                val_preds_prob.extend(probs.cpu().numpy())
                val_targets.extend(y_b.numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds_prob)
        val_acc = accuracy_score(val_targets, np.round(val_preds_prob))

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 记录与调度
        visualizer.update(epoch+1, avg_loss, val_auc, val_acc)
        scheduler.step(val_auc)
        
        # 早停与保存
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_filename)
            print(f"  >>> 模型优化，已保存至 {model_filename}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发! Best Val AUC: {best_auc:.4f}")
                break
    
    # 训练结束后保存图片
    # 此处会输出整个训练过程的 Loss, AUC, Acc 曲线
    visualizer.save_plot(f"models/training_curve_attn_seq{SEQ_LEN}.png")
    print(f"训练结束，总耗时: {time.time() - start_time:.2f}s")

    # --- 5. 最终测试 ---
    print("\n" + "="*50)
    print(">>> 正在加载最佳模型进行测试...")
    
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
    else:
        print("警告: 未找到模型文件，使用当前参数进行测试。")
    
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(DEVICE)
            outputs = model(X_b)
            probs = torch.sigmoid(outputs)
            final_preds.extend(probs.cpu().numpy())
            final_targets.extend(y_b.numpy())
            
    final_preds = np.array(final_preds).flatten()
    final_targets = np.array(final_targets).flatten()
    final_preds_bin = np.round(final_preds)
    
    print("="*50)
    print(f"Attention-GRU (Seq_Len={SEQ_LEN}) 最终测试集表现")
    print("="*50)
    print(f"AUC Score: {roc_auc_score(final_targets, final_preds):.4f}")
    print(f"Accuracy : {accuracy_score(final_targets, final_preds_bin):.4f}")
    print("\n分类报告:")
    print(classification_report(final_targets, final_preds_bin, digits=4))

    torch.cuda.empty_cache()

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == '__main__':
    # Windows 下多进程 DataLoader 必须在 main block 中运行
    run_pipeline()