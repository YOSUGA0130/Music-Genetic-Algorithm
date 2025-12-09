# genetic_algorithm.py
"""
遗传算法
输入：初始旋律种群
经过遗传进化 最多M次迭代
输出：符合适应度要求的旋律数组
"""

import random
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import math
from random_melody import generate_random_melody
from audio_synth import synthesize_melody
from note_encoding import int_to_note

# 编码常量（和 note_encoding.py 对齐）
TIE_CODE = -1 #延长符
REST_CODE = 0 #休止符
MIN_NOTE_CODE = 33   # F3
MAX_NOTE_CODE = 59  # G5

NUM_BARS = 4
UNITS_PER_BAR = 8

# 音乐变换参数
MAX_TRANSFORM_ATTEMPTS = 10  # 音乐变换生成音符超出范围比例达到阈值OUT_OF_RANGE_THRESHOLD后，再次尝试变换的最大尝试次数
TRANSPOSITION_SEMITONE_RANGE = 7  # 移调的半音范围：[-7, 7]（排除0）
INVERSION_AXIS_SCOPE = 2  # 倒影轴的选择范围：平均音高±2
OUT_OF_RANGE_THRESHOLD = 0.1  # 可接受的超出范围音符比例阈值（10%）

# ================= LSTM 模型配置 (必须与训练时一致) =================
MODEL_PATH = "lstm_24.pth"  # 请确保这个文件存在
BATCH_SIZE = 8192       # 暴力拉大，让 GPU 一次吞掉更多数据
EMBEDDING_DIM = 512     # 翻倍，捕捉更细腻的音高语义
HIDDEN_DIM = 2048       # 核心！从 1024 -> 2048，参数量增加 4 倍
NUM_LAYERS = 6          # 增加深度，从 4 -> 6，学习更复杂的乐句结构
DROPOUT = 0.2           # 模型大了，Dropout 保持住
LEARNING_RATE = 0.001   # 大 Batch 通常配合稍大的 LR，0.001 依然安全
EPOCHS = 50              # 多训练几个 Epoch，模型容量大了需要更多时间收敛
DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

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

def fitness_batch(melodies_array: np.ndarray) -> np.ndarray:
    """
    输入: (N, 32) Numpy Array
    输出: (N, ) Numpy Array
    """
    if melodies_array.size == 0 or model is None:
        return np.zeros(len(melodies_array))
    
    all_scores = []
    total_count = len(melodies_array)
    
    model.eval()
    
    with torch.no_grad():
        for i in range(0, total_count, BATCH_SIZE):
            # 1. 切片 (Numpy切片是视图，极快)
            mini_batch = melodies_array[i : min(i + BATCH_SIZE, total_count)]
            
            # 2. 极速编码 (Vectorized)
            encoded_idx = vocab.encode_batch_numpy(mini_batch)
            
            # 3. 转 Tensor (GPU)
            batch_tensor = torch.from_numpy(encoded_idx).to(DEVICE) # from_numpy 共享内存
            
            # 4. 推理
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]
            
            logits = model(inputs) 
            
            # 5. Loss
            loss_fn = nn.CrossEntropyLoss(reduction='none') 
            loss_flat = loss_fn(logits.reshape(-1, len(vocab)), targets.reshape(-1))
            loss_matrix = loss_flat.view(len(mini_batch), -1)
            nll_batch = loss_matrix.mean(dim=1)
            
            # 6. 转回 CPU
            # exp(-nll) 可以直接在 GPU 上算完再转回来，省一点 CPU 算力
            scores_tensor = torch.exp(-nll_batch)
            all_scores.append(scores_tensor.cpu().numpy())
                
    return np.concatenate(all_scores)
    

def run(alpha: float = 0.5,
        m = 200,
        n = 10000, 
        crossover_probability = 0.1,
        mutation_probability = 0.1,
        # 变换概率...
        transposition_probability = 0.05,
        retrograde_probability = 0.05,
        inversion_probability = 0.05,
        retrograde_inversion_probability = 0.03):

    if model is None: return []
    print(f"🚀 High-Performance GA Start (Pop: {n})...")

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


def roulette_wheel_selection(n, fitness_values: List[int]) -> int:
    """
    轮盘赌选择:个体 iℓ 被选中作为亲本的概率等于
    f(iℓ) / Σ(k=1 to N) f(ik)

    input:
        适应度列表
    Returns:
        被选中的个体序号
    """

    # 计算选择概率
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness

    # 轮盘赌选择
    selected_idx = np.random.choice(n, p=probabilities)

    return selected_idx

def crossover(parent1: List[int], parent2: List[int]) -> tuple[List[int], List[int]]:
    """
    随机确定起始位置 start 和交换长度 length，对两个 parent 执行片段交换。
    返回两个子代。
    """
    n = len(parent1)
    assert n == len(parent2)

    # 1. 随机起始位置
    start = random.randint(0, n - 1)

    # 2. 随机交换长度（至少 1）
    max_len = n - start
    length = random.randint(1, max_len)

    end = start + length

    # 3. 复制 parent
    child1 = parent1.copy()
    child2 = parent2.copy()

    # 4. 交换片段
    child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]

    return child1, child2


def mutation_fast(mel_arr: np.ndarray) -> np.ndarray:
    # 这里的 mel_arr 是 (32,) 的 numpy 数组
    # 直接修改它 (in-place) 或者 copy 都可以
    # 上面调用时已经 copy 过了
    pos = random.randint(0, 31)
    
    # 逻辑同原版，只是操作的是 array
    if random.random() < 0.01: # rest_prob
        mel_arr[pos] = REST_CODE
        return mel_arr
        
    can_be_tie = (pos > 0 and mel_arr[pos - 1] != REST_CODE)
    if can_be_tie and random.random() < 0.1: # tie_prob
        mel_arr[pos] = TIE_CODE
        return mel_arr
        
    mel_arr[pos] = random.randint(MIN_NOTE_CODE, MAX_NOTE_CODE)
    return mel_arr

# ============================ 音乐变换操作 ============================


def _apply_and_check_transform(mel: List[int], transform_func) -> Tuple[List[int], int, int]:
    """
    应用音乐变换并检查超出范围的音符数量
    
    参数：
    - mel: 原始旋律
    - transform_func: 变换函数，接受音符值返回新音符值
    
    返回：
    - (变换后的旋律, 超出范围数量, 总音符数量)
    
    注意：
    - 超出范围的音符会随机变成休止符(0)或延长符(-1)
    - 第一个位置不能是延长符，会强制变成休止符
    - 延长符只能跟在音符或延长符后面，不能跟在休止符后面
    """
    new_mel = mel.copy()
    out_of_range_count = 0
    note_count = 0
    
    for i in range(len(new_mel)):
        # 跳过休止符和延长符
        if new_mel[i] <= 0:
            continue
        
        note_count += 1
        # 应用变换函数
        new_note = transform_func(new_mel[i])
        
        # 检查是否在有效范围内
        if MIN_NOTE_CODE <= new_note <= MAX_NOTE_CODE:
            new_mel[i] = new_note
        else:
            # 超出范围：随机变成休止符或延长符
            # TODO: 会导致最后的的结果全部都是0！！！
            out_of_range_count += 1
            # if random.random() < 0.5: 
            #     new_mel[i] = REST_CODE
            # else:
            #     new_mel[i] = TIE_CODE
            
    # 确保第一个位置不是延长符
    if len(new_mel) > 0 and new_mel[0] == TIE_CODE:
        new_mel[0] = REST_CODE
    
    # 确保延长符不跟在休止符后面
    for i in range(1, len(new_mel)):
        if new_mel[i] == TIE_CODE and new_mel[i - 1] == REST_CODE:
            new_mel[i] = REST_CODE
    
    return new_mel, out_of_range_count, note_count


def _is_transform_acceptable(out_of_range_count: int, note_count: int) -> bool:
    """
    检查变换结果是否可接受
    
    参数：
    - out_of_range_count: 超出范围的音符数量
    - note_count: 总音符数量
    
    返回：
    - bool: True表示可接受，False表示不可接受
    """
    threshold = OUT_OF_RANGE_THRESHOLD
    if note_count == 0:
        return True
    return out_of_range_count / note_count <= threshold


def transposition(mel: List[int], semitone_range: int = TRANSPOSITION_SEMITONE_RANGE) -> List[int]:
    """
    将整个旋律的音高整体升高或降低若干半音
    
    参数：
    - semitone_range: [-semitone_range, semitone_range](排除0)是随机移调的最大半音数范围
                      默认使用全局常量TRANSPOSITION_SEMITONE_RANGE

    返回：
    - 移调后的旋律数组
    
    注意：
    - 休止符(REST_CODE=0)和延长符(TIE_CODE=-1)保持不变
    - 超出范围的音符会随机变成休止符(0)或延长符(-1)
    - 第一个位置不能是延长符，会强制变成休止符
    - 延长符只能跟在音符或延长符后面，不能跟在休止符后面
    - 如果超过OUT_OF_RANGE_THRESHOLD的音符超出范围，重新移调，最多尝试MAX_TRANSFORM_ATTEMPTS次，如果都不行，返回原旋律
    """
    
    # 生成可用的移调范围，排除0
    available_semitones = [i for i in range(-semitone_range, semitone_range + 1) if i != 0]
    
    for _ in range(MAX_TRANSFORM_ATTEMPTS):
        # 随机选择移调幅度
        semitones = random.choice(available_semitones)        # 定义移调变换函数
        def transpose_note(note: int) -> int:
            return note + semitones
        
        # 应用变换并检查
        new_mel, out_of_range_count, note_count = _apply_and_check_transform(mel, transpose_note)
        
        # 如果超出范围的音符不超过10%，接受这个移调
        if _is_transform_acceptable(out_of_range_count, note_count):
            return new_mel
    
    # 如果10次尝试都不行，返回原旋律
    return mel.copy()


def retrograde(mel: List[int]) -> List[int]:
    """
    直接反转旋律列表
    
    
    参数：
    - mel: 旋律编码数组
    
    返回：
    - 逆行后的旋律数组（时间反向）
    
    
    注意：
    - 不是反转音符，没有打包音符和音符后面的延长符一起反转
    - 第一个位置不能是延长符，会自动转换为休止符
    - 延长符只能跟在音符或延长符后面，不能跟在休止符后面
    """
    # 直接反转整个列表
    reversed_mel = mel[::-1]
    
    # 修正第一个位置：如果是延长符，改为休止符
    if reversed_mel[0] == TIE_CODE:
        reversed_mel[0] = REST_CODE
    
    # 修正"延长符在休止符后"的情况
    for i in range(1, len(reversed_mel)):
        if reversed_mel[i] == TIE_CODE and reversed_mel[i - 1] == REST_CODE:
            # 延长符不能跟在休止符后面，改为休止符
            reversed_mel[i] = REST_CODE
    
    return reversed_mel


def inversion(mel: List[int], scope: int = INVERSION_AXIS_SCOPE) -> List[int]:
    """
    将旋律以某个音高为轴进行镜像反转
    
    参数：
    - scope: 在随机选择轴时，围绕平均音高的范围±scope内选择轴
             默认使用全局常量INVERSION_AXIS_SCOPE
    
    返回：
    - 倒影后的旋律数组
    
    注意：
    - 休止符(0)和延长符(-1)保持不变
    - 超出范围的音符会随机变成休止符(0)或延长符(-1)
    - 第一个位置不能是延长符，会强制变成休止符
    - 延长符只能跟在音符或延长符后面，不能跟在休止符后面
    - 如果超过OUT_OF_RANGE_THRESHOLD的音符超出范围，重新随机选择轴，最多尝试MAX_TRANSFORM_ATTEMPTS次，如果都不行，返回原旋律
    """
    # 提取所有有效音符
    notes = [n for n in mel if n > 0]
    if len(notes) == 0:
        return mel.copy()
    
    # 计算音符范围，用于确定合适的轴范围
    min_note = min(notes)
    max_note = max(notes)
    avg_note = (min_note + max_note) // 2
    axis_min = max(MIN_NOTE_CODE, avg_note - scope)
    axis_max = min(MAX_NOTE_CODE, avg_note + scope)
    
    for _ in range(MAX_TRANSFORM_ATTEMPTS):
        # 在平均音高附近±scope的范围内随机选择轴
        axis = random.randint(axis_min, axis_max)
        
        # 定义倒影变换函数
        def invert_note(note: int) -> int:
            return 2 * axis - note
        
        # 应用变换并检查
        new_mel, out_of_range_count, note_count = _apply_and_check_transform(mel, invert_note)
        
        # 如果超出范围的音符不超过10%，接受这个倒影
        if _is_transform_acceptable(out_of_range_count, note_count):
            return new_mel
    
    # 如果10次尝试都不行，返回原旋律
    return mel.copy()


def retrograde_inversion(mel: List[int]) -> List[int]:
    """
    逆行倒影复合变换
    
    参数：
    - axis: 倒影的中心轴音高，传递给inversion函数
            如果为None，则使用旋律中第一个非休止/延长音符作为轴
    
    返回：
    - 逆行倒影后的旋律数组
    
    """
    # 先倒影，再逆行
    inverted = inversion(mel)
    return retrograde(inverted)



if __name__ == "__main__":
    result = run(alpha=0.3, m=100, n=2000, crossover_probability=0.3, mutation_probability=0.5,)
    print(result)
    for idx, mel in enumerate(result[:10]):
        output_path = f"./output/genetic_algorithm_result_{idx+1}.wav"
        synthesize_melody(
            codes=mel,
            output_path=output_path,
            sample_dir="./samples/",
            BPM=167,
            unit_time=180  
        )

        int_to_note(mel, f"./output/genetic_algorithm_result_{idx+1}.json")



