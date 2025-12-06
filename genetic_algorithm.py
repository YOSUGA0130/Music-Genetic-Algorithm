# genetic_algorithm.py
"""
遗传算法
输入：初始旋律种群
经过遗传进化 最多M次迭代
输出：符合适应度要求的旋律数组
"""

import random
import numpy as np
from typing import List
from random_melody import generate_random_melody

# 编码常量（和 note_encoding.py 对齐）
TIE_CODE = -1 #延长符
REST_CODE = 0 #休止符
MIN_NOTE_CODE = 33   # F3
MAX_NOTE_CODE = 59  # G5

NUM_BARS = 4
UNITS_PER_BAR = 8

def fitness(mel: List[int]) -> float:
    # 适应度函数计算
    pass

def run(alpha: float = 0.5,
        m =100,
        n = 10,
        crossover_probability = 0.1,
        mutation_probability = 0.1):
    """
    运行遗传算法

    alpha: 适应度阈值
    m:  迭代次数
    n: 种群大小
    crossover_probability: 交叉概率
    mutation_probability: 变异概率
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
                # 其他（移调，倒影........）待补充
                pass
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
