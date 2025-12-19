from typing import List, Tuple
from config import *

# ================= 配置常量 =================

# 钢琴键 A0 (21) 对应的 offset
# Code 1 = A0 (Midi 21)
# 我们的输入 Code -> Midi 公式: midi = code + 20
CODE_TO_MIDI_OFFSET = 20

# 禁用的首尾音 (Pitch Class)
# A = 9, F = 5 (基于 C=0, C#=1...)
# 解释: A0=21. 21%12 = 9 (A). F1=29. 29%12 = 5 (F).
FORBIDDEN_START_END_PITCH_CLASSES = {5, 9} 

# 协和音程 (半音数)
# 0(纯1), 3(小3), 4(大3), 7(纯5), 8(小6), 9(大6), 12(纯8)
CONSONANT_INTERVALS = {0, 3, 4, 7, 8, 9, 12, 15, 16, 19, 24}

# 不协和/不稳定音程 (半音数) - 主要是 2度, 三全音, 7度
DISSONANT_INTERVALS = {1, 2, 6, 10, 11, 13, 14, 22, 23}

# 强拍位置 (假设 4/4 拍, 每小节 8 个单位)
# 强拍: 0, 4, 8, 12... (即每 4 个单位是一个拍点)
BEAT_STEP = 4 

def get_pitch_class(code: int) -> int:
    """计算音符的音名类别 (0-11, C=0)"""
    midi = code + CODE_TO_MIDI_OFFSET
    return midi % 12

def get_real_notes_with_indices(mel: List[int]):
    """
    预处理旋律，提取实际发声的音符及其位置。
    忽略休止符，将延音符(-1)归并到前一个音符的时值中(此处简化为只取音高)。
    返回: List of (index, code)
    """
    notes = []
    current_note = None
    
    for i, code in enumerate(mel):
        if code > 0:
            # 新的音符
            current_note = code
            notes.append((i, code))
        elif code == TIE_CODE:
            # 延音
            pass
        elif code == REST_CODE:
            # 休止符
            current_note = None
            
    return notes

def calculate_interval_score(note1: int, note2: int) -> float:
    """
    计算两个音符之间音程的适应度分数 (0.0 - 1.0)
    """
    interval = abs(note1 - note2)
    
    # 1. 惩罚过大音程 (大于八度)
    if interval > 12:
        # 距离越远分越低，非线性下降
        return 0
        
    # 2. 奖励协和音程
    if interval in CONSONANT_INTERVALS:
        return 1.0
        
    # 3. 惩罚不协和音程
    if interval in DISSONANT_INTERVALS:
        return 0.3  # 允许少量不协和作为经过音，但不鼓励
        
    return 0.6  # 其他中性音程 (如纯4度在某些语境下)

def fitness(mel: List[int]) -> float:
    """
    适应度函数
    
    规则来源：https://oss.wanfangdata.com.cn/www/基于遗传算法与人工神经网络的二声部创意曲自动生成.ashx?isread=true&type=degree&resourceId=Y1344370&transaction=%7B%22id%22%3Anull%2C%22transferOutAccountsStatus%22%3Anull%2C%22transaction%22%3A%7B%22id%22%3A%221997714390224347136%22%2C%22status%22%3A1%2C%22createDateTime%22%3Anull%2C%22payDateTime%22%3A1765127201025%2C%22authToken%22%3A%22TGT-2023328-DfMYzUQOFVVXZ13jucjtaNsAFpgWl0CpKje6VeeHy0pjJfSJpe-auth-iploginservice-85d6db8c6b-5jbjr%22%2C%22user%22%3A%7B%22accountType%22%3A%22Group%22%2C%22key%22%3A%22北京大学%22%7D%2C%22transferIn%22%3A%7B%22accountType%22%3A%22Income%22%2C%22key%22%3A%22ThesisFulltext%22%7D%2C%22transferOut%22%3A%7B%22GTimeLimit.北京大学%22%3A30.0%7D%2C%22turnover%22%3A30.0%2C%22orderTurnover%22%3A30.0%2C%22productDetail%22%3A%22degree_Y1344370%22%2C%22productTitle%22%3Anull%2C%22userIP%22%3A%22106.120.124.118%22%2C%22organName%22%3Anull%2C%22memo%22%3Anull%2C%22orderUser%22%3A%22北京大学%22%2C%22orderChannel%22%3A%22pc%22%2C%22payTag%22%3A%22IPV6%22%2C%22webTransactionRequest%22%3Anull%2C%22signature%22%3A%22QjUCGmsYrW%2F6okTCEfBBSoZeYu%2FM0pmeY%2F37DCMRJzQZ%2FYGYytApUvCFuF%2BZEVN%2FkObKTzr%2Fjtoh%5Cnn9MH6Nld4j8NlyanzafJw0gTtaDr4Xpcd642Dk3az27BCR8rcUz1sqLK79r1SyRQ0U5GZtyERv6K%5CnKewEsiSr4ZjzKVnKQt4%3D%22%7D%2C%22isCache%22%3Afalse%7D
    1. 禁止 F 和 A 在首尾出现 (F/A 为不稳定音)。
    2. 强拍位置的音符体现主和弦印象 (强拍之间应和谐)。
    3. 音程不过大，且尽量和谐。
    
    返回: 0.0 (最差) ~ 1.0 (最好)
    """
    if not mel or all(x <= 0 for x in mel):
        return 0.0

    # 提取有效音符序列 (index, code)
    real_notes = get_real_notes_with_indices(mel)
    if not real_notes:
        return 0.0

    # weights (总和 1.0)
    w_structure = 0.1  # 首尾规则权重
    w_melodic   = 0.6  # 旋律线条流畅/和谐权重
    w_harmonic  = 0.3  # 骨干音(强拍)和谐权重

    # ================= 1. 结构分数 (Start/End Constraints) =================
    start_note_code = real_notes[0][1]
    end_note_code = real_notes[-1][1]
    
    start_pc = get_pitch_class(start_note_code)
    end_pc = get_pitch_class(end_note_code)
    
    # 检查首尾是否包含 F(5) 或 A(9)
    # 规则：若包含则扣分。完全遵守得 1.0，违反一个得 0.5，都违反得 0.0
    structure_score = 1.0
    if start_pc in FORBIDDEN_START_END_PITCH_CLASSES:
        structure_score -= 0.5
    if end_pc in FORBIDDEN_START_END_PITCH_CLASSES:
        structure_score -= 0.5
    
    structure_score = max(0.0, structure_score)

    # ================= 2. 旋律音程分数 (Melodic Interval) =================
    # 连续音符之间不要有大跳，要和谐
    interval_scores = []
    if len(real_notes) > 1:
        for i in range(len(real_notes) - 1):
            n1 = real_notes[i][1]
            n2 = real_notes[i+1][1]
            # 计算相邻两个实际音符的音程分
            interval_scores.append(calculate_interval_score(n1, n2))
        
        melodic_score = sum(interval_scores) / len(interval_scores)
    else:
        melodic_score = 1.0 # 只有一个音，完美平滑

    # ================= 3. 骨干音/强拍和谐分数 (Harmonic Stability) =================
    # 强拍和次强拍音符最有意义，体现主和弦
    # 策略：提取所有落在强拍上的音符，计算它们两两相邻的音程是否和谐
    strong_beat_notes = []
    
    # 构建一个查找表: index -> code
    mel_dict = {idx: code for idx, code in real_notes}
    
    # 遍历所有可能的强拍位置 (0, 4, 8, 12...)
    for i in range(0, len(mel), BEAT_STEP):
        # 只有当该位置是音符时才算作骨干音
        if mel[i] > 0:
            strong_beat_notes.append(mel[i])
        elif mel[i] == TIE_CODE:
            # 往回找它属于哪个音
            k = i - 1
            while k >= 0:
                if mel[k] > 0:
                    strong_beat_notes.append(mel[k])
                    break
                elif mel[k] == REST_CODE:
                    break 
                k -= 1
    
    beat_interval_scores = []
    if len(strong_beat_notes) > 1:
        for i in range(len(strong_beat_notes) - 1):
            n1 = strong_beat_notes[i]
            n2 = strong_beat_notes[i+1]
            score = calculate_interval_score(n1, n2)
            beat_interval_scores.append(score)
        
        harmonic_score = sum(beat_interval_scores) / len(beat_interval_scores)
    else:
        harmonic_score = 0.5 

    # ================= 总分计算 =================
    total_fitness = (
        structure_score * w_structure +
        melodic_score * w_melodic +
        harmonic_score * w_harmonic
    )

    return round(total_fitness, 4)
