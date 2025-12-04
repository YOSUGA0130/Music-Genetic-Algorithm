# audio_synth.py

"""
根据整数编码数组合成音频。

用法示例：
    from audio_synth import synthesize_melody

    codes = [40, 42, 44, 0, 44, 42, 40, -1, 47, ...]  # 一串编码
    synthesize_melody(
        codes,
        output_path="output.wav",
        sample_dir="samples",
        unit_time=250  # 八分音符的时长，例: 120BPM 下八分音符 ≈ 250ms
    )
"""

import os
from typing import List
from pydub import AudioSegment
from note_encoding import note_to_int


TIE_CODE = -1  #延长符
REST_CODE = 0  #休止符


def synthesize_melody(
    codes: List[int],
    output_path: str,
    sample_dir: str = "samples",
    BPM: int = 120,
    unit_time: int = 155
) -> None:
    """
    根据整数编码数组合成音频并导出为 wav 文件。

    参数：
    - codes : List[int]
        旋律编码数组（-1 延长, 0 休止, 1-88 钢琴键）
        每个元素占用一个八分音符时值
    - output_path : str
        导出的 wav 文件路径，例如 "/Melody/Audio/output.wav"
    - sample_dir : str
        存放样本 wav 的文件夹路径，里面应有 "1.wav" ~ "88.wav"
    - BPM : 拍子数(乐谱头会注明)
    - unit_time : int
        每一个数组元素(八分音符)对应的实际时间长度(ms)
        unit_time = 30000/BPM (ms)
    """

    if not codes:
        raise ValueError("codes 为空，无法合成音频")

    # 预加载用到的样本，避免重复读盘
    sample_cache: dict[int, AudioSegment] = {}
    max_sample_len = 0
    # unit_time = round(30000/BPM)
    # 其实可以输入BPM再转成unit_time，但实操下来感觉不如直接输入八分音符时值(

    used_note_codes = sorted({c for c in codes if c > 0})
    if not used_note_codes:
        # 全部是休止/延长，直接输出一段静音
        total_dur = len(codes) * unit_time
        silence = AudioSegment.silent(duration=total_dur)
        silence.export(output_path, format="wav")
        return

    for code in used_note_codes:
        sample_path = os.path.join(sample_dir, f"{code}.wav")
        if not os.path.isfile(sample_path):
            raise FileNotFoundError(
                f"找不到样本文件: {sample_path}（需要 {code}.wav）"
            )
        seg = AudioSegment.from_wav(sample_path)
        sample_cache[code] = seg
        if len(seg) > max_sample_len:
            max_sample_len = len(seg)

    # 最终音频至少要覆盖：
    # 所有单位格的时间 + 最长样本尾部
    base_duration = len(codes) * unit_time + max_sample_len
    output = AudioSegment.silent(duration=base_duration)

    # 逐个格子叠加样本
    for idx, code in enumerate(codes):
        if code <= 0:
            # -1（延长）或 0（休止）都不触发新的音符
            continue

        if code not in sample_cache:
            # 理论上不会发生，因为前面已经用 used_note_codes 预加载
            continue

        note_seg = sample_cache[code]
        start_time_ms = idx * unit_time

        # 在指定时间位置叠加（mix），保留前后音的重叠
        output = output.overlay(note_seg, position=start_time_ms)

    # 导出为 wav 文件
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output.export(output_path, format="wav")


if __name__ == "__main__":
    codes = note_to_int("Melody/Code/test.json")
    synthesize_melody(codes, "Melody/Audio/test.wav", sample_dir="samples", unit_time=180)
    pass