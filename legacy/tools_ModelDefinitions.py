"""
模块名称：模型结构定义 (Model Definitions)
功能：统一存储项目中使用的 PyTorch 模型类定义，确保训练与回测时的结构完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ResNet-MLP (对应 05_ResNet_GPU.py)
# ==============================================================================
class ResNetMLP(nn.Module):
    """
    基于 LayerNorm 和 Bottleneck 结构的残差网络。
    输入: (Batch, Features)
    输出: (Batch, 1) -> Logits (未经过 Sigmoid)
    """
    def __init__(self, input_dim):
        super(ResNetMLP, self).__init__()
        
        # 输入映射
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 残差块 1 (等宽: 128 -> 128)
        self.res_block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        # 残差块 2 (Bottleneck: 128 -> 64 -> 128)
        self.res_block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )
        
        # 输出层 (无 Sigmoid)
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        
        # Block 1
        identity = x
        out = self.res_block1(x)
        out += identity
        out = self.relu(out)
        
        # Block 2
        identity = out
        out = self.res_block2(out)
        out += identity
        out = self.relu(out)
        
        return self.output_layer(out)


# ==============================================================================
# 2. GRU / Attention 组件 (对应 06_GRU.py)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, gru_output):
        # gru_output: (Batch, Seq, Hidden)
        scores = self.attention_layer(gru_output)
        weights = F.softmax(scores, dim=1)
        return torch.sum(gru_output * weights, dim=1)

class GRUWithAttention(nn.Module):
    """
    带 Attention 的单向 GRU 模型。
    输入: (Batch, Seq, Features)
    输出: (Batch, 1) -> Probabilities (经过 Sigmoid)
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GRUWithAttention, self).__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=False
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() # GRU 模型通常自带 Sigmoid
        )

    def forward(self, x):
        out, _ = self.gru(x)
        context = self.attention(out)
        return self.fc(context)

class CNN_GRU_Model(nn.Module):
    """
    1D-CNN + GRU 混合模型。
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(CNN_GRU_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Feat, Seq)
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), 0.3)
        x = x.permute(0, 2, 1) # (Batch, Seq, Feat)
        out, _ = self.gru(x)
        context = self.bn2(self.attention(out))
        return self.fc(context)