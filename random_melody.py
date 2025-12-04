# random_melody.py
"""
生成一个简单的随机旋律（编码数组）。


要求：
- 拍号:4/4
- 最小单位：八分音符
- 每小节 8 个八分音符
- 共 4 小节 -> 4 * 8 = 32 个单位
- 乐音体系从F3 -> G5 , 对应数字是33-59, 加上 0 和 -1 ，只在这个范围内生成

  返回一条长度为 32 的整数数组。
"""

import random
from typing import List
from note_encoding import int_to_note
from audio_synth import synthesize_melody

# 编码常量（和 note_encoding.py 对齐）
TIE_CODE = -1 #延长符
REST_CODE = 0 #休止符
MIN_NOTE_CODE = 33   # F3
MAX_NOTE_CODE = 59  # G5

NUM_BARS = 4
UNITS_PER_BAR = 8


def generate_random_melody(
    num_bars: int = NUM_BARS,
    units_per_bar: int = UNITS_PER_BAR,
    rest_probability: float = 0.1,
    tie_probability: float = 0.2,
) -> List[int]:
    """
    生成一条随机旋律的编码数组。

    参数：
    - num_bars : 小节数，默认 4
    - units_per_bar : 每小节的单位数，默认 8(即8个八分音符)
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
    if not (0.0 <= rest_probability <= 1.0):
        raise ValueError("rest_probability 必须在 [0, 1] 范围内")
    if not (0.0 <= tie_probability <= 1.0):
        raise ValueError("tie_probability 必须在 [0, 1] 范围内")

    total_units = num_bars * units_per_bar
    melody: List[int] = []

    for i in range(total_units):
        if i == 0:
            # 第一个位置不能是延长符，只能是休止或音符
            r = random.random()
            if r < rest_probability:
                code = REST_CODE
            else:
                code = random.randint(MIN_NOTE_CODE, MAX_NOTE_CODE)
            melody.append(code)
            continue

        prev = melody[i - 1]

        # 优先考虑是否生成延长符：
        # 只有当前一个不是休止符时才有资格成为延长符
        if prev != REST_CODE:
            r_tie = random.random()
            if r_tie < tie_probability:
                melody.append(TIE_CODE)
                continue

        # 否则：在这一格生成“休止 or 新音符”
        r_rest = random.random()
        if r_rest < rest_probability:
            code = REST_CODE
        else:
            code = random.randint(MIN_NOTE_CODE, MAX_NOTE_CODE)

        melody.append(code)

    return melody


if __name__ == "__main__":
    #直接运行该程序，会生成对应json和audio
    mel = generate_random_melody()
    print(mel)

    int_to_note(mel, "Melody/Code/test.json") #保存json音名文件
    synthesize_melody(mel, "Melody/Audio/test.wav", sample_dir="samples", unit_time=180)