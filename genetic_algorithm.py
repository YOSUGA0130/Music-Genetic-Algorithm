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
from random_melody import generate_random_melody

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


def fitness(mel: List[int]) -> float:
    # 适应度函数计算
    pass

def run(alpha: float = 0.5,
        m =100,
        n = 10,
        crossover_probability = 0.1,
        mutation_probability = 0.1,
        transposition_probability = 0.05,
        retrograde_probability = 0.05,
        inversion_probability = 0.05,
        retrograde_inversion_probability = 0.03):
    """
    运行遗传算法

    alpha: 适应度阈值
    m:  迭代次数
    n: 种群大小
    crossover_probability: 交叉概率
    mutation_probability: 变异概率
    transposition_probability: 移调概率
    retrograde_probability: 逆行概率
    inversion_probability: 倒影概率
    retrograde_inversion_probability: 逆行倒影概率
    """

    # 1. 初始化种群
    random_generate_melodies = [generate_random_melody() for _ in range(n)]

    population = random_generate_melodies.copy()

    # 2. 迭代 m 次
    for _ in range(m):
        # 2.1 计算适应度
        fitnesses = [fitness(mel) for mel in population]

        # 2.2 是否存在 ≥ α 的适应度的个体
        good_fitness_mel = [mel for mel in population if fitness(mel) >= alpha]

        # 2.3 进化生成下一代，直到下一代数目达到N
        next_generation = []
        # 复制：适应度 ≥ α 的个体直接进入下一代
        next_generation.extend(good_fitness_mel)
        # 适应度 < α 的个体通过交叉、变异等生成下一代
        while len(next_generation) < n:
            k = random.random() # 决定进行什么进化操作
            if k < crossover_probability:
                # 交换
                selected_indices = [roulette_wheel_selection(n, fitnesses) for _ in range(2)]
                parent1, parent2 = population[selected_indices[0]], population[selected_indices[1]]
                child1, child2 = crossover(parent1, parent2)
                next_generation.extend([child1, child2])
            elif k < crossover_probability + mutation_probability:
                # 变异
                selected_idx = roulette_wheel_selection(n, fitnesses)
                child = mutation(population[selected_idx])
                next_generation.append(child)
            else:
                # 其他音乐变换操作：移调、逆行、倒影、逆行倒影
                selected_idx = roulette_wheel_selection(n, fitnesses)
                parent = population[selected_idx]
                
                # 随机选择一种音乐变换
                transform_choice = random.random()
                transform_sum = (transposition_probability + retrograde_probability + 
                               inversion_probability + retrograde_inversion_probability)
                
                if transform_sum > 0:
                    # 归一化概率
                    t_prob = transposition_probability / transform_sum
                    r_prob = retrograde_probability / transform_sum
                    i_prob = inversion_probability / transform_sum
                    
                    if transform_choice < t_prob:
                        # 移调
                        child = transposition(parent)
                    elif transform_choice < t_prob + r_prob:
                        # 逆行
                        child = retrograde(parent)
                    elif transform_choice < t_prob + r_prob + i_prob:
                        # 倒影
                        child = inversion(parent)
                    else:
                        # 逆行倒影
                        child = retrograde_inversion(parent)
                    
                    next_generation.append(child)
                else:
                    # 如果所有变换概率都为0，直接复制父代
                    next_generation.append(parent.copy())
        population = next_generation[:n]

    # 3. 返回迭代后适应度 ≥ α 的旋律们 (按照适应度从大到小排列)
    result = [mel for mel in population if fitness(mel) >= alpha].sort(key=fitness, reverse=True)

    if result == None:
        print("没有找到适应度大于等于α的旋律")
    else:
        print("最高适应度：", fitness(result[0]))

    return result



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

def crossover(parent1: List[int], parent2: List[int]) -> (List[int], List[int]):
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


def mutation(
    mel: List[int],
    rest_probability: float = 0.1,
    tie_probability: float = 0.2,
) -> List[int]:
    """
    随机选择一个位置进行变异，产生一个新个体

    参数：
    - rest_probability : 每个单位成为休止符的概率(0~1)
    - tie_probability  : 在允许的情况下, 每个单位成为延长符的概率(0~1)

    约束：
    - 任意位置可以是休止符（按 rest_probability 随机）
    - 延长符 -1 只能出现在 “非首个位置 且 前一个不是休止符” 的地方：
        即前一个必须是音符 (1-88) 或延长符 (-1)

    返回：
    - List[int]，长度 = num_bars * units_per_bar
      每个元素是 -1, 0 或 1..88
    """

    new_mel = mel.copy()
    pos = random.randint(0, len(mel) - 1)

    # --- 1. 尝试生成休止符 ---
    if random.random() < rest_probability:
        new_mel[pos] = REST_CODE
        return new_mel

    # --- 2. 尝试生成延长符 ---
    can_be_tie = (
        pos > 0 and           # 不能是第一个
        new_mel[pos - 1] != REST_CODE  # 前一个不能是休止符
    )

    if can_be_tie and random.random() < tie_probability:
        new_mel[pos] = TIE_CODE
        return new_mel

    # --- 3. 否则生成普通音符 ---
    new_mel[pos] = random.randint(MIN_NOTE_CODE, MAX_NOTE_CODE)
    return new_mel


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
            out_of_range_count += 1
            if random.random() < 0.5: 
                new_mel[i] = REST_CODE
            else:
                new_mel[i] = TIE_CODE
            
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
