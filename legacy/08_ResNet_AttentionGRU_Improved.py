"""
程序名称：08_ResNet_Attention_GRU_Improved.py
功能描述：基于 ResNet-Attention-GRU 的混合神经网络模型（改进版）

改进点 (相较于 07 版):
1. [架构升级] ResNet 特征宽度增加：将 ResNet 特征提取层的维度从 64 提升至 128，增强模型对微观结构特征的捕获能力。
2. [策略优化] 学习率调度升级：采用 CosineAnnealingWarmRestarts (带热重启的余弦退火) 替代原有的 ReduceLROnPlateau。
   这有助于模型在训练后期跳出局部最优解，寻找更好的全局解。

模型原理 (Hybrid Architecture):
   [Input Sequence] (Batch, 20, 13)
          ⬇
   [TimeDistributed ResNet (Dim=128)] 对每个时间步独立进行更深度的特征提取
          ⬇
   [High-Level Features] (Batch, 20, 128)
          ⬇
   [Bi-GRU Layer (Hidden=128)] 双向捕捉时序演变
          ⬇
   [Attention Mechanism] 自动加权关键时间步
          ⬇
   [Classifier] 全连接层 -> Logits
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
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import gc
import time

# ==============================================================================
# 0. 全局配置区
# ==============================================================================
TRAIN_FILE = 'ready_train.parquet'
TEST_FILE = 'ready_test.parquet'
MODEL_SAVE_PATH = 'models/best_hybrid_resnet128_gru_improved.pth'

# 模型超参数
SEQ_LEN = 20           # 时间窗口
LEARNING_RATE = 1e-3   # 初始学习率
EPOCHS = 200           # Cosine策略需要更多轮次来发挥效果
PATIENCE = 30          # 容忍度适当调大
DROPOUT_RATE = 0.3     # 保持较高的 Dropout 防止过拟合
WEIGHT_DECAY = 1e-4    # 保持 L2 正则化

# 硬件配置
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    # 由于 ResNet 变宽，显存占用增加，适当调整 Batch Size
    BATCH_SIZE = 8192      
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    # print(f">> CUDA 环境检测成功 ({torch.cuda.get_device_name(0)})。")
else:
    DEVICE = torch.device('cpu')
    BATCH_SIZE = 2048
    NUM_WORKERS = 0
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False
    print(">> 使用 CPU/MPS 环境。")

if not os.path.exists('models'):
    os.makedirs('models')

# ==============================================================================
# 1. 数据集与工具类 (保持不变)
# ==============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, seq_len):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.seq_len = seq_len
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len], self.labels[idx + self.seq_len - 1]

class TrainingVisualizer:
    def __init__(self):
        self.epochs, self.losses, self.val_aucs, self.val_accs, self.lrs = [], [], [], [], []
    def update(self, epoch, loss, val_auc, val_acc, lr):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.val_aucs.append(val_auc)
        self.val_accs.append(val_acc)
        self.lrs.append(lr)
    def save_plot(self, filename):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot Loss & Metrics
        color = 'tab:red'
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.epochs, self.losses, color=color, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Metrics', color=color)
        ax2.plot(self.epochs, self.val_aucs, color=color, marker='s', label='Val AUC')
        ax2.plot(self.epochs, self.val_accs, color='green', marker='^', linestyle='--', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Training Metrics')
        ax1.grid(True, alpha=0.3)

        # Plot LR Schedule
        ax3.plot(self.epochs, self.lrs, color='purple', linestyle='-')
        ax3.set_title('Learning Rate Schedule (Cosine Annealing)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename); plt.close()

# ==============================================================================
# 2. 混合模型定义 (保持结构不变，参数在初始化时调整)
# ==============================================================================

# --- 组件 1: ResNet 基础块 (用于特征提取) ---
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

# --- 组件 2: 注意力机制 ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, gru_output):
        # gru_output: (Batch, Seq, Hidden)
        energy = self.attn(gru_output)          # (Batch, Seq, 1)
        weights = F.softmax(energy, dim=1)      # (Batch, Seq, 1)
        context = torch.sum(weights * gru_output, dim=1) # (Batch, Hidden)
        return context, weights

# --- 核心模型: ResNet + Attention + GRU ---
class ResNetAttentionGRU(nn.Module):
    def __init__(self, input_dim, resnet_dim=64, gru_dim=128, num_gru_layers=2):
        super(ResNetAttentionGRU, self).__init__()
        
        # 1. 特征提取部分 (Time-Distributed ResNet)
        # 先将输入映射到高维空间
        self.input_mapping = nn.Sequential(
            nn.Linear(input_dim, resnet_dim),
            nn.LayerNorm(resnet_dim),
            nn.ReLU()
        )
        # ResNet 块进行深层特征交互
        self.resnet_block = ResNetBlock(resnet_dim, dropout=DROPOUT_RATE)
        
        # 2. 时序建模部分 (Bi-Directional GRU)
        # 输入维度是 ResNet 的输出维度
        self.gru = nn.GRU(
            input_size=resnet_dim,
            hidden_size=gru_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=DROPOUT_RATE,
            bidirectional=True  # 双向能利用未来信息反推当前状态
        )
        
        # 3. 注意力部分
        # 双向 GRU 的输出维度是 gru_dim * 2
        self.attention = Attention(gru_dim * 2)
        
        # 4. 分类器部分
        self.fc = nn.Sequential(
            nn.Linear(gru_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1) # Logits
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        batch_size, seq_len, _ = x.size()
        
        # --- Step 1: TimeDistributed Feature Extraction ---
        # 展平时间维度以并行通过 ResNet: (Batch * Seq, Input_Dim)
        x_flat = x.reshape(-1, x.size(2))
        
        # 通过 ResNet 提取特征
        x_flat = self.input_mapping(x_flat)
        x_flat = self.resnet_block(x_flat)
        
        # 恢复时间维度: (Batch, Seq, ResNet_Dim)
        x_seq = x_flat.reshape(batch_size, seq_len, -1)
        
        # --- Step 2: GRU Sequence Modeling ---
        # gru_out: (Batch, Seq, GRU_Dim * 2)
        gru_out, _ = self.gru(x_seq)
        
        # --- Step 3: Attention Mechanism ---
        # context: (Batch, GRU_Dim * 2)
        context, _ = self.attention(gru_out)
        
        # --- Step 4: Classification ---
        logits = self.fc(context)
        return logits

# ==============================================================================
# 3. 训练流程 (核心修改区域)
# ==============================================================================
def run_pipeline():
    print(f"启动改进版混合模型训练 (ResNet-128 + CosineAnnealing)...")
    start_time = time.time()

    # --- 数据加载 ---
    print("加载数据...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    # 过滤特征
    target_features = [
        'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
        'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
        'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
    ]
    cols = [c for c in target_features if c in train_df.columns]
    input_dim = len(cols)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[cols].values)
    X_test = scaler.transform(test_df[cols].values)
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values
    
    del train_df, test_df; gc.collect()

    # --- DataLoader ---
    split = int(len(X_train) * 0.8)
    train_ds = TimeSeriesDataset(X_train[:split], y_train[:split], SEQ_LEN)
    val_ds = TimeSeriesDataset(X_train[split:], y_train[split:], SEQ_LEN)
    test_ds = TimeSeriesDataset(X_test, y_test, SEQ_LEN)

    kwargs = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': PIN_MEMORY}
    if PERSISTENT_WORKERS: kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    
    print(f"数据准备完毕。Device: {DEVICE}")

    # --- 模型初始化 [改进点1：增加 ResNet 宽度] ---
    print("初始化模型 (ResNet Dim: 128, GRU Dim: 128)...")
    # 将 resnet_dim 从 64 提升到 128
    model = ResNetAttentionGRU(input_dim=input_dim, resnet_dim=128, gru_dim=128).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    # --- 调度器初始化 [改进点2：使用余弦退火热重启] ---
    # T_0=15: 第一个周期为 15 Epoch
    # T_mult=2: 后续周期加倍 (15 -> 30 -> 60 ...)
    print("初始化调度器 (CosineAnnealingWarmRestarts)...")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    visualizer = TrainingVisualizer()
    best_auc = 0
    patience_cnt = 0

    # --- 训练循环 ---
    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x)
                probs.extend(torch.sigmoid(out).cpu().numpy())
                targets.extend(y.numpy())
        
        val_auc = roc_auc_score(targets, probs)
        val_acc = accuracy_score(targets, np.round(probs))
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        visualizer.update(epoch+1, avg_loss, val_auc, val_acc, current_lr)
        
        # [改进点2注意] CosineAnnealing 需要在每个 Epoch 结束时更新，而不是基于验证指标
        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> 模型优化，已保存: {MODEL_SAVE_PATH}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            # Cosine 策略下，Loss 可能会在重启时反弹，因此早停需要更宽容
            if patience_cnt >= PATIENCE and current_lr < 1e-5: 
                print(f"早停触发 (Patience={PATIENCE} & LR too low)。Best AUC: {best_auc:.4f}")
                break

    visualizer.save_plot("models/training_curve_hybrid_improved.png")
    print(f"总耗时: {time.time()-start_time:.1f}s")

    # --- 最终测试 ---
    print("\n>>> 最终测试集评估...")
    if os.path.exists(MODEL_SAVE_PATH): 
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"已加载最优模型权重: {MODEL_SAVE_PATH}")
    
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            all_probs.extend(torch.sigmoid(out).cpu().numpy())
            all_targets.extend(y.numpy())
    
    auc = roc_auc_score(all_targets, all_probs)
    preds = np.round(all_probs)
    print("="*50)
    print(f"Improved Hybrid Model Test AUC : {auc:.4f}")
    print(f"Improved Hybrid Model Test Acc : {accuracy_score(all_targets, preds):.4f}")
    print(classification_report(all_targets, preds, digits=4))

if __name__ == '__main__':
    run_pipeline()