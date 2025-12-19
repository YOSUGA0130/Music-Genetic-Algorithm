import torch
import torch.nn as nn
import numpy as np
from config import *

# ================= 1. 定义词表类 (与训练代码一致) =================
class MusicVocab:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.rest_token = 0
        self.tie_token = -1
        
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        self._add_token(self.pad_token)  # 0
        self._add_token(self.rest_token) # 1
        self._add_token(self.tie_token)  # 2
        for pitch in range(1, 101):
            self._add_token(pitch)
            
        # === 新增：构建向量化查找表 (Lookup Table) ===
        # 假设最大 MIDI code 不超过 128 (通常 1-100 够了)
        max_code = 128
        self.lookup_table = np.full(max_code, self.token_to_idx[self.rest_token], dtype=np.int64)
        
        # 填充查找表
        for code, idx in self.token_to_idx.items():
            if isinstance(code, int) and 0 <= code < max_code:
                self.lookup_table[code] = idx
        # 特殊处理负数 (TIE_CODE = -1)
        # 这种 trick 可以让我们直接处理 -1: 将其映射到 lookup_table 的最后一个位置或其他方式
        # 这里为了简单，我们单独处理负数，或者将 -1 视为特殊索引
        self.tie_idx = self.token_to_idx[self.tie_token]

    def _add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
    
    def encode_batch_numpy(self, batch_arr: np.ndarray) -> np.ndarray:
        """
        极速编码：直接对 Numpy 数组进行操作
        输入: (N, 32) int array (包含 -1, 0, 33-59)
        输出: (N, 32) int array (包含 0-N indices)
        """
        # 1. 创建结果数组，默认填充 Rest Index
        encoded = np.full_like(batch_arr, self.token_to_idx[self.rest_token], dtype=np.int64)
        
        # 2. 处理普通音符和休止符 (>=0 的值)
        # 使用 mask 提取 >=0 的部分进行查表
        mask_positive = batch_arr >= 0
        # 这里的 clip 是为了防止偶尔产生的异常大数值导致越界
        safe_values = np.clip(batch_arr, 0, len(self.lookup_table)-1)
        encoded[mask_positive] = self.lookup_table[safe_values[mask_positive]]
        
        # 3. 处理延长符 (-1)
        mask_tie = (batch_arr == TIE_CODE)
        encoded[mask_tie] = self.tie_idx
        
        return encoded

    # 保留旧方法兼容单条处理
    def encode(self, seq):
        return [self.token_to_idx.get(x, self.token_to_idx[self.rest_token]) for x in seq]
    
    def decode(self, idx_seq):
        """把 Index 数组转回原始旋律"""
        return [self.idx_to_token.get(x, 0) for x in idx_seq]
    
    def __len__(self):
        return len(self.token_to_idx)

# ================= 2. 定义模型类 (与训练代码一致) =================
class MelodyLSTM_God(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(MelodyLSTM_God, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 你的 3090 喜欢大矩阵运算
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # LayerNorm 有助于大模型收敛
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 映射回词表
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        # 加入 LayerNorm 和 Dropout
        out = self.ln(lstm_out)
        out = self.dropout_layer(out)
        
        logits = self.fc(out)
        return logits
