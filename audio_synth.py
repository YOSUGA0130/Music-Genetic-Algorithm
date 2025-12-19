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
    根据音乐片段数组合成音频并导出为 wav 文件。
    output_path : 导出的 wav 文件路径，
    BPM : tempo
    unit_time : 每一个音符对应的实际时间长度(ms)
    unit_time = 30000/BPM (ms)
    """

    if not codes:
        raise ValueError("无法合成音频")

    sample_cache: dict[int, AudioSegment] = {}
    max_sample_len = 0
    # unit_time = round(30000/BPM)
    # 其实可以输入BPM再转成unit_time，但实操下来感觉不如直接输入八分音符时值(

    used_note_codes = sorted({c for c in codes if c > 0})
    if not used_note_codes:
        #全是休止或者延长就输出静音
        total_dur = len(codes) * unit_time
        silence = AudioSegment.silent(duration=total_dur)
        silence.export(output_path, format="wav")
        return

    for code in used_note_codes:
        sample_path = os.path.join(sample_dir, f"{code}.wav")
        if not os.path.isfile(sample_path):
            raise FileNotFoundError(
                f"找不到钢琴琴键: {sample_path}（需要 {code}.wav）"
            )
        seg = AudioSegment.from_wav(sample_path)
        sample_cache[code] = seg
        if len(seg) > max_sample_len:
            max_sample_len = len(seg)

    base_duration = len(codes) * unit_time + max_sample_len
    output = AudioSegment.silent(duration=base_duration)

    for idx, code in enumerate(codes):
        if code <= 0:
            continue

        if code not in sample_cache:
            continue

        note_seg = sample_cache[code]
        start_time_ms = idx * unit_time

        output = output.overlay(note_seg, position=start_time_ms)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output.export(output_path, format="wav")


if __name__ == "__main__":
    codes = note_to_int("Melody/Code/test.json")
    synthesize_melody(codes, "Melody/Audio/test.wav", sample_dir="samples", unit_time=180)
    pass