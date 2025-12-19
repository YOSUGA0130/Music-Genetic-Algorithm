import random
from typing import List
from note_encoding import int_to_note
from audio_synth import synthesize_melody


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

    num_bars : 小节数，默认 4
    units_per_bar : 每小节8个八分音符
    rest_probability : 每个单位成为休止符的概率(0~1)
    tie_probability  : 在允许的情况下, 每个单位成为延长符的概率(0~1)

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

        if prev != REST_CODE:
            r_tie = random.random()
            if r_tie < tie_probability:
                melody.append(TIE_CODE)
                continue

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