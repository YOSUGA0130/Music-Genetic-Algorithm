
import random
from typing import List

# 编码常量（和 note_encoding.py 对齐）
TIE_CODE = -1 #延长符
REST_CODE = 0 #休止符
MIN_NOTE_CODE = 33   # F3
MAX_NOTE_CODE = 59  # G5

NUM_BARS = 4
UNITS_PER_BAR = 8

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
