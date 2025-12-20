# fitness_lstm.py
"""
lstm 适应度模块
"""

import random
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
from population import generate_random_melody
from util.audio_synth import synthesize_melody
from util.note_encoding import int_to_note
from config import *
from genetics import (MAX_TRANSFORM_ATTEMPTS, TRANSPOSITION_SEMITONE_RANGE, INVERSION_AXIS_SCOPE, OUT_OF_RANGE_THRESHOLD,
                      roulette_wheel_selection, crossover, _apply_and_check_transform, _is_transform_acceptable,
                      transposition, retrograde, inversion, retrograde_inversion)
from fitness_rule_enhance import fitness_enhanced

# ================= LSTM 模型配置 (必须与训练时一致) =================
MODEL_PATH = "model/lstm_24.pth"  # 请确保这个文件存在
BATCH_SIZE = 8192       # 暴力拉大，让 GPU 一次吞掉更多数据
EMBEDDING_DIM = 512     # 翻倍，捕捉更细腻的音高语义
HIDDEN_DIM = 2048       # 核心！从 1024 -> 2048，参数量增加 4 倍
NUM_LAYERS = 6          # 增加深度，从 4 -> 6，学习更复杂的乐句结构
DROPOUT = 0.2           # 模型大了，Dropout 保持住
LEARNING_RATE = 0.001   # 大 Batch 通常配合稍大的 LR，0.001 依然安全
EPOCHS = 50              # 多训练几个 Epoch，模型容量大了需要更多时间收敛
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ================= 3. 全局加载模型 (只加载一次) =================
print(f"正在加载 LSTM 评分模型: {MODEL_PATH} ...")
try:
    # 初始化词表
    vocab = MusicVocab()
    
    # 初始化模型结构
    model = MelodyLSTM_God(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    
    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # ================= 核心修复代码开始 =================
    # 检测是否包含 torch.compile 产生的 "_orig_mod." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            # 去掉前缀 "_orig_mod." (长度为10)
            name = k[10:] 
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # ================= 核心修复代码结束 =================
    
    # 加载清洗后的字典
    model.load_state_dict(new_state_dict)
    
    model.to(DEVICE)
    model.eval() # 开启评估模式 (关闭 Dropout)
    print("模型加载成功！")
    
except Exception as e:
    print(f"严重错误: 无法加载模型 ({e})。请确保 {MODEL_PATH} 存在且配置与训练时一致。")
    model = None

# ================= 4. 并行版 Fitness 函数 (核心修改) =================

def _calc_penalty(n):
    """
    计算连续休止或延长的惩罚系数
    1次连续：1.0（无惩罚）
    2次连续：0.9（10%惩罚）
    3-4次连续：0.7（30%惩罚）
    5次连续：0.7 × 0.5 = 0.35
    6次连续：0.7 × 0.5² = 0.175
    ...
    """
    if n <= 1: return 1.0
    if n == 2: return 0.9
    if n <= 4: return 0.7
    return 0.7 * (0.5 ** (n - 4))

def batch_structure_penalty(melodies_array):
    """
    批量计算结构惩罚
    """
    N, L = melodies_array.shape
    penalties = np.ones(N, dtype=np.float32)
    
    for i in range(N):
        melody = melodies_array[i]
        p = 1.0
        cur_run = 0
        cur_type = None # REST_CODE or TIE_CODE
        
        for j in range(L):
            val = melody[j]
            if val == REST_CODE or val == TIE_CODE:
                if val == cur_type:
                    cur_run += 1
                else:
                    if cur_run > 1:
                        p *= _calc_penalty(cur_run)
                    cur_type = val
                    cur_run = 1
            else:
                if cur_run > 1:
                    p *= _calc_penalty(cur_run)
                cur_type = None
                cur_run = 0
        
        if cur_run > 1:
            p *= _calc_penalty(cur_run)
            
        penalties[i] = p
    return penalties

def fitness_batch(melodies_array: np.ndarray) -> np.ndarray:
    """
    输入: (N, 32) Numpy Array
    输出: (N, ) Numpy Array
    
    混合评分策略：
    1. 第一小节 (0-7): 使用 fitness_rule_enhance 进行规则评分 (权重 0.25)
    2. 后三小节 (8-31): 使用 LSTM 模型评分 + 结构惩罚 (权重 0.75)
    """
    if melodies_array.size == 0 or model is None:
        return np.zeros(len(melodies_array))
    
    total_count = len(melodies_array)
    
    # === Part 1: 第一小节规则评分 ===
    # 提取第一小节 (N, 8)
    part_rule = melodies_array[:, :8]
    rule_scores = np.zeros(total_count, dtype=np.float32)
    
    # 由于 fitness_enhanced 是单体函数，这里使用循环
    # 考虑到性能，如果 N 很大，这里可能是瓶颈，但规则计算通常比 LSTM 快
    for i in range(total_count):
        # 转为 list 传给 fitness_enhanced
        # num_bars=1 表示只评估这一个小节
        rule_scores[i] = fitness_enhanced(part_rule[i].tolist(), num_bars=1)
        
    # === Part 2: 后三小节 LSTM 评分 (带上下文) ===
    # 关键修改：输入完整旋律，但只计算后三小节的 Loss
    
    all_lstm_scores = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, total_count, BATCH_SIZE):
            # 1. 切片 (取完整旋律)
            mini_batch = melodies_array[i : min(i + BATCH_SIZE, total_count)]
            
            # 2. 编码 (完整 32 长度)
            encoded_idx = vocab.encode_batch_numpy(mini_batch)
            
            # 3. 转 Tensor
            batch_tensor = torch.from_numpy(encoded_idx).to(DEVICE)
            
            # 4. 推理
            # 输入: [0...30], 目标: [1...31]
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]
            
            logits = model(inputs) 
            
            # 5. Loss 计算 (带 Mask)
            loss_fn = nn.CrossEntropyLoss(reduction='none') 
            
            # 计算所有位置的 Loss: (Batch, 31)
            # 注意：logits 是 (Batch, 31, Vocab), targets 是 (Batch, 31)
            # 为了使用 CrossEntropyLoss，我们需要 reshape
            loss_flat = loss_fn(logits.reshape(-1, len(vocab)), targets.reshape(-1))
            loss_matrix = loss_flat.view(len(mini_batch), -1) # (Batch, 31)
            
            # 关键：只取后 24 个时间步的 Loss (对应后三小节)
            # 索引 0-6 对应第一小节的预测 (预测第 1-7 个音)，索引 7 对应预测第 8 个音
            # 我们需要评估的是从第 8 个音开始的预测准确性
            # targets 的索引：
            # idx 0: target is note[1] (bar 1)
            # ...
            # idx 6: target is note[7] (bar 1 end)
            # idx 7: target is note[8] (bar 2 start) -> 这是我们要开始评估的第一个点
            
            # 所以我们取 loss_matrix[:, 7:]
            relevant_loss = loss_matrix[:, 7:]
            
            # 计算平均 NLL
            nll_batch = relevant_loss.mean(dim=1)
            
            # 6. 转回 CPU
            scores_tensor = torch.exp(-nll_batch)
            all_lstm_scores.append(scores_tensor.cpu().numpy())
                
    lstm_raw_scores = np.concatenate(all_lstm_scores)
    
    # === Part 3: 结构惩罚 (仅针对 LSTM 部分) ===
    # 计算后三小节的结构惩罚
    part_lstm = melodies_array[:, 8:] # 依然只对后三小节计算惩罚
    struct_penalties = batch_structure_penalty(part_lstm)
    
    # LSTM 部分最终得分
    lstm_final_scores = lstm_raw_scores * struct_penalties
    
    # === Part 4: 综合评分 ===
    # 权重分配：1小节 vs 3小节 -> 0.25 : 0.75
    final_scores = 0.25 * rule_scores + 0.75 * lstm_final_scores

    return final_scores * 2.0
    

def run(alpha: float = 0.5,
        m = 200,
        n = 10000, 
        crossover_probability = 0.1,
        mutation_probability = 0.25, 
        # 变换概率...
        transposition_probability = 0.1,
        retrograde_probability = 0.05,
        inversion_probability = 0.05,
        retrograde_inversion_probability = 0.03):

    if model is None: return []
    print(f"High-Performance GA Start (Pop: {n})...")

    # 1. 初始化种群
    initial_pop = [generate_random_melody(rest_probability=0.01, tie_probability=0.1) for _ in range(n)]
    population = np.array(initial_pop, dtype=np.int32)

    # === 修复开始：分开计算总概率和相对概率 ===
    # 1. 计算所有变换操作的总概率 (用于主层级分配)
    total_transform_prob = (transposition_probability + retrograde_probability + 
                            inversion_probability + retrograde_inversion_probability)
    
    # 2. 计算变换内部的相对概率 (用于子层级分配，归一化到1)
    transform_relative_probs = np.array([
        transposition_probability, retrograde_probability, 
        inversion_probability, retrograde_inversion_probability
    ])
    if transform_relative_probs.sum() > 0:
        transform_relative_probs = transform_relative_probs / transform_relative_probs.sum()
    # === 修复结束 ===

    for generation in range(m):
        # 2.1 批量计算适应度
        fitnesses = fitness_batch(population)
        
        # 统计信息
        best_idx = np.argmax(fitnesses)
        current_best = fitnesses[best_idx]
        print(f"Gen {generation}: Max Fitness = {current_best:.6f}")

        # 2.2 精英保留 (Elitism)
        sorted_indices = np.argsort(fitnesses)[::-1] 
        elite_count = int(n * 0.1)
        elite_indices = sorted_indices[:elite_count]
        
        next_population = np.empty((n, 32), dtype=np.int32)
        next_population[:elite_count] = population[elite_indices]
        
        # 2.3 生成剩余个体
        slots_needed = n - elite_count
        
        # === 修复点：使用 total_transform_prob (0.18) 而不是归一化后的和 (1.0) ===
        probs = [crossover_probability, mutation_probability, total_transform_prob]
        
        # 计算剩余概率 (Copy)
        remaining_prob = 1.0 - sum(probs)
        # 防止浮点误差导致的微小负数 (-1e-17)
        if remaining_prob < 0: remaining_prob = 0.0
        
        probs.append(remaining_prob)
        
        # 再次归一化防止浮点误差报错 (sum可能不严格等于1.0)
        probs = np.array(probs)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # 极端情况防崩溃
            probs = np.array([0, 0, 0, 1.0]) 
        
        # 随机决定每个 slot 的命运
        # 0: Crossover, 1: Mutation, 2: Transform, 3: Copy
        op_codes = np.random.choice(4, size=slots_needed, p=probs)
        
        # 向量化选择父母
        fit_sum = fitnesses.sum()
        if fit_sum > 0:
            select_probs = fitnesses / fit_sum
        else:
            select_probs = np.ones(n) / n
            
        parents_indices = np.random.choice(n, size=(slots_needed, 2), p=select_probs)
        
        # 2.4 执行批量操作
        cursor = elite_count
        
        # (A) 批量交叉
        cross_mask = (op_codes == 0)
        n_cross = np.sum(cross_mask)
        if n_cross > 0:
            p1_idx = parents_indices[cross_mask, 0]
            p2_idx = parents_indices[cross_mask, 1]
            p1_pop = population[p1_idx]
            p2_pop = population[p2_idx]
            
            split_points = np.random.randint(1, 32, size=n_cross)
            col_indices = np.arange(32)
            mask = col_indices < split_points[:, None]
            children = np.where(mask, p1_pop, p2_pop)
            
            next_population[cursor : cursor + n_cross] = children
            cursor += n_cross

        # (B) 批量变异
        mut_mask = (op_codes == 1)
        n_mut = np.sum(mut_mask)
        if n_mut > 0:
            p_idx = parents_indices[mut_mask, 0]
            targets = population[p_idx].copy()
            for i in range(n_mut):
                targets[i] = mutation_fast(targets[i])
            next_population[cursor : cursor + n_mut] = targets
            cursor += n_mut

        # (C) 批量变换
        trans_mask = (op_codes == 2)
        n_trans = np.sum(trans_mask)
        if n_trans > 0:
            p_idx = parents_indices[trans_mask, 0]
            targets = population[p_idx]
            
            # === 修复点：这里使用 transform_relative_probs ===
            sub_ops = np.random.choice(4, size=n_trans, p=transform_relative_probs)
            
            processed = []
            for i in range(n_trans):
                t_type = sub_ops[i]
                orig = targets[i]
                if t_type == 0: res = transposition(orig)
                elif t_type == 1: res = retrograde(orig)
                elif t_type == 2: res = inversion(orig)
                else: res = retrograde_inversion(orig)
                processed.append(res)
            
            next_population[cursor : cursor + n_trans] = np.array(processed)
            cursor += n_trans

        # (D) 剩余复制
        copy_mask = (op_codes == 3)
        n_copy = np.sum(copy_mask)
        if n_copy > 0:
            p_idx = parents_indices[copy_mask, 0]
            next_population[cursor : cursor + n_copy] = population[p_idx]
            cursor += n_copy
            
        population = next_population

    # 3. 最终排序
    final_fitness = fitness_batch(population)
    sort_idx = np.argsort(final_fitness)[::-1]
    
    mask_pass = final_fitness[sort_idx] >= alpha
    pass_indices = sort_idx[mask_pass]
    
    if len(pass_indices) == 0:
        print("未找到满足阈值的旋律，返回最佳。")
        result_pop = population[sort_idx[:1]] 
    else:
        result_pop = population[pass_indices]
        print(f"最高适应度：{final_fitness[sort_idx[0]]:.6f}")

    return result_pop.tolist()


def mutation_fast(mel_arr: np.ndarray) -> np.ndarray:
    # 这里的 mel_arr 是 (32,) 的 numpy 数组
    num_mutations = random.randint(1, 3)
    
    for _ in range(num_mutations):
        pos = random.randint(0, 31)
        
        # 逻辑同原版，只是操作的是 array
        if random.random() < 0.05: # rest_prob 稍微提高一点
            mel_arr[pos] = REST_CODE
            continue
            
        can_be_tie = (pos > 0 and mel_arr[pos - 1] != REST_CODE)
        if can_be_tie and random.random() < 0.1: # tie_prob
            mel_arr[pos] = TIE_CODE
            continue
            
        mel_arr[pos] = random.randint(MIN_NOTE_CODE, MAX_NOTE_CODE)
        
    return mel_arr

