"""
程序名称：基于 ResNet-MLP 的高频交易方向预测模型 (优化版)

1. 程序概述：
   本程序实现了一个结合了残差连接（ResNet）思想的多层感知机（MLP）深度学习模型。
   专门针对表格型（Tabular）高频微观结构数据进行设计，预测未来3秒的价格变动方向。
   
   相比于传统 MLP，引入残差连接有助于缓解深层网络中的梯度消失问题。
   本版本针对 NVIDIA RTX 4090 等高端 GPU 进行了 I/O 和计算效率的专项优化。

2. 输入数据：
   - 训练集：'ready_train.parquet'
   - 测试集：'ready_test.parquet'
   - 特征维度：13维微观结构特征。
   - 标签维度：二分类 (0: 下跌, 1: 上涨)。

3. 模型结构 (ResNetMLP)：
   - 输入层：线性映射 -> 128维。
   - 残差块：y = f(x) + x 结构，使用 LayerNorm 替代 BatchNorm 以适应大 Batch 训练。
   - 输出层：线性输出 (Logits)，配合 BCEWithLogitsLoss 提升数值稳定性。

4. 功能特性：
   - 平台自适应：自动识别 Windows(CUDA) 与 Mac/CPU 环境，调整 BatchSize 与 Workers。
   - 静态绘图：训练结束后保存 Loss/AUC 曲线图至 models/ 目录。
   - 性能优化：使用 pin_memory 和 persistent_workers 加速数据吞吐。
   - 早停机制：防止过拟合，自动保存验证集表现最好的模型。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，只保存不显示
import matplotlib.pyplot as plt
import time
import os
import platform

# ==============================================================================
# 0. 全局配置区 (平台自适应)
# ==============================================================================
TRAIN_FILE = 'ready_train.parquet'
TEST_FILE = 'ready_test.parquet'
LEARNING_RATE = 5e-3
EPOCHS = 100
PATIENCE = 15

# 检测运行环境并配置硬件参数
if torch.cuda.is_available():
    # --- Windows / Linux with NVIDIA GPU (如 RTX 4090) ---
    DEVICE = torch.device('cuda')
    BATCH_SIZE = 32768      # 4090 显存大，极大增加 Batch Size 提升吞吐
    NUM_WORKERS = 12        # CPU 多线程加载
    PIN_MEMORY = True       # 锁页内存，加速 CPU->GPU 传输
    PERSISTENT_WORKERS = True
    # print(f">> 检测到 CUDA 环境 ({torch.cuda.get_device_name(0)})，已启用高性能配置。")
else:
    # --- Mac (MPS) / CPU ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        # print(">> 检测到 Mac MPS 环境，使用标准配置。")
    else:
        DEVICE = torch.device('cpu')
        # print(">> 检测到 CPU 环境，使用标准配置。")
    
    BATCH_SIZE = 4096       # Mac/CPU 内存共享，不宜过大
    NUM_WORKERS = 0         # Mac 上多进程有时不稳定，设为 0
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
    逻辑：
        1. 记录每个 Epoch 的 Loss, AUC, Accuracy。
        2. save_plot() 方法绘制双轴图表并保存为 PNG 文件。
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

        plt.title('ResNet-MLP Training Metrics')
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        plt.close()
        print(f"训练曲线已保存至: {filename}")

class LOBDataset(Dataset):
    """
    功能：PyTorch 数据集封装。
    
    输入：特征矩阵 X (numpy array), 标签向量 y (numpy array)。
    输出：支持通过索引 idx 获取 (tensor_x, tensor_y)。
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1) # (N, 1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==============================================================================
# 2. 模型结构定义 (优化版)
# ==============================================================================

class ResNetMLP(nn.Module):
    """
    功能：基于残差连接的多层感知机模型。
    
    结构优化：
    1. 使用 LayerNorm 代替 BatchNorm (在大 Batch 下表现更稳，且不仅依赖 batch 统计量)。
    2. 移除输出层 Sigmoid，配合 BCEWithLogitsLoss 使用。
    """
    def __init__(self, input_dim):
        super(ResNetMLP, self).__init__()
        
        # 特征映射层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),  # 优化：改用 LayerNorm
            nn.ReLU()
        )
        
        # 残差块 1 (等宽)
        self.res_block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        # 残差块 2 (Bottleneck 结构)
        self.res_block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128), # 升维回 128
            nn.LayerNorm(128)
        )
        
        # 输出层 (输出 Logits，无 Sigmoid)
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 移除 nn.Sigmoid()，由 Loss 函数处理
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 输入映射
        x = self.input_layer(x)
        
        # 残差连接 1
        identity = x
        out = self.res_block1(x)
        out += identity
        out = self.relu(out)
        
        # 残差连接 2
        identity = out
        out = self.res_block2(out) 
        out += identity
        out = self.relu(out)
        
        # 输出 Logits
        return self.output_layer(out)

# ==============================================================================
# 3. 训练与评估流程
# ==============================================================================

def train_model():
    """
    功能：执行完整的数据加载、模型初始化、训练循环与评估流程。
    """
    print("正在加载数据...")
    df_train = pd.read_parquet(TRAIN_FILE)
    df_test = pd.read_parquet(TEST_FILE)
    
    # 特征选择
    target_features = [
        'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
        'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
        'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
    ]
    feature_cols = [c for c in target_features if c in df_train.columns]
    
    X = df_train[feature_cols].values
    y = df_train['Label'].values
    
    X_test_final = df_test[feature_cols].values
    y_test_final = df_test['Label'].values
    
    # 划分验证集 (8:2)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_final = scaler.transform(X_test_final)
    
    # 构建 DataLoader (使用优化后的参数)
    # 对于 CUDA 环境，启用 num_workers 和 pin_memory
    train_loader = DataLoader(
        LOBDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        LOBDataset(X_val, y_val), 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        LOBDataset(X_test_final, y_test_final), 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"数据准备完毕: Train={X_train.shape}, Val={X_val.shape}, Test={X_test_final.shape}")
    print(f"当前配置: BatchSize={BATCH_SIZE}, Workers={NUM_WORKERS}, Device={DEVICE}")
    
    # 初始化模型与优化器
    model = ResNetMLP(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 优化点：使用 BCEWithLogitsLoss 替代 BCELoss
    criterion = nn.BCEWithLogitsLoss()
    
    # 初始化可视化与早停变量
    visualizer = TrainingVisualizer()
    best_auc = 0
    no_improve_count = 0
    save_path = 'models/best_resnet_mlp.pth' 
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch) # Outputs are logits
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_preds_prob, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(DEVICE)
                outputs = model(X_b)
                # 由于模型去掉了 Sigmoid，这里评估时需要手动加上以获得概率
                probs = torch.sigmoid(outputs)
                val_preds_prob.extend(probs.cpu().numpy())
                val_targets.extend(y_b.numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds_prob)
        val_acc = accuracy_score(val_targets, np.round(val_preds_prob))
        
        # 记录数据
        visualizer.update(epoch+1, avg_loss, val_auc, val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # --- 早停与保存 ---
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path) 
            print(f"  >>> 模型优化，已保存至 {save_path}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                print(f"早停触发! Best Val AUC: {best_auc:.4f}")
                break
    
    print(f"训练结束，总耗时: {time.time() - start_time:.2f}s")
    
    # 保存训练曲线图
    visualizer.save_plot("models/training_curve_resnet.png")

    # ==============================================================================
    # 4. 最终测试 (使用 Test Set)
    # ==============================================================================
    print("\n" + "="*50)
    print("正在加载最佳模型并在测试集上进行最终评估...")
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        print("警告：未找到保存的模型文件，使用当前最后一次迭代的模型进行测试。")

    model.eval()
    
    final_preds, final_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(DEVICE))
            probs = torch.sigmoid(outputs) # 同样手动加 Sigmoid
            final_preds.extend(probs.cpu().numpy())
            final_targets.extend(y_batch.numpy())
            
    final_preds = np.array(final_preds).flatten()
    final_targets = np.array(final_targets).flatten()
    final_preds_bin = np.round(final_preds)
    
    print("="*50)
    print("ResNet-MLP (Optimized) 最终测试集表现")
    print("="*50)
    print(f"AUC Score: {roc_auc_score(final_targets, final_preds):.4f}")
    print(f"Accuracy : {accuracy_score(final_targets, final_preds_bin):.4f}")
    print("\n分类报告:")
    print(classification_report(final_targets, final_preds_bin, digits=4))

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == '__main__':
    # Windows 下多进程 DataLoader 必须在 if __name__ == '__main__': 保护块中运行
    train_model()