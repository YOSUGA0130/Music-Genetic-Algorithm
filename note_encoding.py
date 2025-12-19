import json
import re
from typing import List

# 88 键的 MIDI 范围
START_MIDI = 21   # A0
END_MIDI = 108    # C8

REST_CODE = 0
TIE_CODE = -1

#音名
PITCH_CLASSES = ["C", "#C", "D", "#D", "E", "F",
                 "#F", "G", "#G", "A", "#A", "B"]
PC_TO_INDEX = {pc: i for i, pc in enumerate(PITCH_CLASSES)}


def _note_name_to_midi(note: str) -> int:
    """
    将单个音名转换为midi
    """
    m = re.fullmatch(r"([#b]?)([A-Ga-g])(\d)", note.strip())
    if not m:
        raise ValueError(f"无法解析音名: {note!r}")

    accidental, letter, octave_str = m.groups()
    letter = letter.upper()
    octave = int(octave_str)

    if letter not in PC_TO_INDEX:
        raise ValueError(f"不支持音名: {note!r}")
    base_idx = PC_TO_INDEX[letter]

    # 升降号
    if accidental == "#":
        idx = (base_idx + 1) % 12
    elif accidental == "b":
        idx = (base_idx - 1) % 12
    else:
        idx = base_idx

    midi = 12 * (octave + 1) + idx  # 标准 MIDI 公式

    if not (START_MIDI <= midi <= END_MIDI):
        raise ValueError(
            f"音高 {note!r} (MIDI {midi}) 不在 A0~C8 范围内"
        )

    return midi


def _midi_to_note_name(midi: int) -> str:
    """
    将midi转换为音名字符串, 使用升号。
    """
    if not (START_MIDI <= midi <= END_MIDI):
        raise ValueError(
            f"MIDI {midi} 不在 A0~C8 范围内，无法转换为音名"
        )

    idx = midi % 12
    octave = midi // 12 - 1
    pc = PITCH_CLASSES[idx]
    return f"{pc}{octave}"


def note_to_int(json_path: str) -> List[int]:
    """
    读取音名旋律文件,
    返回数组。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    if not isinstance(tokens, list):
        raise ValueError("JSON 文件内容必须是数组")

    result: List[int] = []

    for token in tokens:
        if not isinstance(token, str):
            raise ValueError(f"数组元素必须是字符串，收到: {token!r}")

        t = token.strip()

        if t == "0":
            result.append(REST_CODE)
        elif t == "-":
            result.append(TIE_CODE)
        else:
            midi = _note_name_to_midi(t)
            code = midi - 20  # 21->1, 108->88
            result.append(code)

    return result


def int_to_note(codes: List[int], json_path: str) -> None:
    """
    读取一个整数数组,传入数组 and 保存json文件的位置和文件名
    转换为音名字符串数组，并保存为 JSON 文件, 默认保存位置/Melody/Code

    """
    tokens: List[str] = []

    for code in codes:
        if code == TIE_CODE:
            tokens.append("-")
        elif code == REST_CODE:
            tokens.append("0")
        else:
            if not (1 <= code <= 88):
                raise ValueError(
                    f"编码 {code} 不在合法音高编码范围内"
                )
            midi = code + 20  # 1->21, 88->108
            note_name = _midi_to_note_name(midi)
            tokens.append(note_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pass
