# fitness_rule_enhance.py
"""
优化后的遗传算法 - 核心评价体系：fitness_enhanced
采用五个维度的评价：
		"temperament": 	# 律学与音程协和度
        "tonality":     # 调式功能
        "pc_set":       # 音类集合色彩
        "rhythm":       # 节奏奇性与活力
        "symmetry":    	# 旋律对称与结构冗余

优先保证调性逻辑，其次是音程的物理协和程度，把局部色彩、节奏与旋律对称相对后置。
使用线性插值方法，让权重随小节数平滑过渡。

支持 1-16 小节旋律生成，每小节 8 个八分音符
"""

import random
import numpy as np
from typing import List, Tuple
import math

# 外部模块引用（请确保路径正确）
from population import generate_random_melody
from util.audio_synth import synthesize_melody
from config import *
from genetics import (
	roulette_wheel_selection,
	crossover,
	mutation,
	transposition,
	retrograde,
	inversion,
	retrograde_inversion
)

# ================= 1. 乐理配置常量 =================

CODE_TO_MIDI_OFFSET = 20
FORBIDDEN_START_END_PITCH_CLASSES = {5, 9}
BEAT_STEP = 4

# ================= 维度一：律学常量配置 =================

# 基于频率比简单程度的评分 (0.0 ~ 1.0)
# 参考 王杰 老师课件: 2:1(八度), 3:2(纯五), 4:3(纯四), 5:4(大三), 6:5(小三)
# 十二平均律下这些音程虽然有微小偏差，但在听觉上被归类为协和
INTERVAL_PURITY_SCORE = {
	0: 1.0,  # 纯一度 (1:1)
	12: 1.0,  # 纯八度 (2:1)
	7: 0.95,  # 纯五度 (3:2) - 极高协和
	5: 0.85,  # 纯四度 (4:3) - 高协和
	4: 0.80,  # 大三度 (5:4) - 协和
	9: 0.75,  # 大六度 (5:3) - 协和
	3: 0.70,  # 小三度 (6:5) - 协和
	8: 0.65,  # 小六度 (8:5) - 协和

	# 不协和音程 (大二、小七、增四等)
	2: 0.30,  # 大二度 (9:8) - 不稳定
	10: 0.25,  # 小七度 (9:5) - 不稳定
	1: 0.10,  # 小二度 (16:15) - 极不协和
	11: 0.10,  # 大七度 (15:8) - 极不协和
	6: 0.05,  # 增四/减五 (三全音 45:32) - "狼"音程趋势，极不稳定
}

# 音分偏差惩罚 (针对十二平均律偏离纯律的补偿)
# 比如大三度在平均律(400音分)比纯律(386音分)高了14音分。。
ET_DEVIATION_PENALTY = {
	4: 0.05,  # 大三度偏高 14 cents
	9: 0.06,  # 大六度偏高 16 cents
	7: 0.01,  # 纯五度偏差极小 (2 cents)
}

# ================= 维度二：调性与功能配置 =================

# 目标音阶：C大调音阶的音类集合 (C, D, E, F, G, A, B)
# 如果想写a小调，改成 {9, 11, 0, 2, 4, 5, 7}
TARGET_SCALE_PCS = {0, 2, 4, 5, 7, 9, 11}

# 音级稳定性评分
# 1.0: 主音 (C) - 极稳
# 0.8: 属音/中音 (G, E) - 稳（主和弦音）
# 0.4: 下属音/上主音 (F, D, A) - 不稳
# 0.1: 导音 (B) - 极不稳，强烈趋向主音
STABILITY_MAP = {
	0: 1.0,  # I
	7: 0.8,  # V
	4: 0.8,  # III
	5: 0.4,  # IV
	2: 0.4,  # II
	9: 0.4,  # VI
	11: 0.1,  # VII
}

# ================= 维度三：音类集合配置 =================

# 目标向量：我们希望旋律的局部片段尽量接近“大/小三和弦”的色彩
# 即使是旋律，局部音符构成的集合若符合此向量，听感也会非常“悦耳”
TARGET_VECTOR = [0, 0, 1, 1, 1, 0]

# 集合类奖惩
# 3-11 (大/小三和弦) 的原型是 {0, 3, 7} 或 {0, 4, 7}
# 4-27 (属七/半减七) 的原型向量是 [0, 1, 2, 1, 1, 1]

# ================= 维度四：节奏活力配置 =================

# 理想密度：每小节 8 个单位中，触发 4~6 个音符（50%~75%）最为平衡
IDEAL_DENSITY = 0.6

# 强弱拍权重 (4/4 拍，8个单位)
# 索引: 0(强) 1(弱) 2(次强) 3(弱) 4(强) 5(弱) 6(次强) 7(弱)
# 为了增加“奇性”，我们有时会奖励弱拍起音（切分音）
METRICAL_WEIGHTS = [1.2, 0.6, 0.9, 0.5, 1.1, 0.5, 0.8, 0.4]


# ================= 2. 基础辅助函数 =================

def get_pitch_class(code: int) -> int:
	"""计算音符的音名类别 (0-11, C=0)"""
	return (code + CODE_TO_MIDI_OFFSET) % 12


def get_real_notes_with_indices(mel: List[int]):
	"""提取实际发声的音符及其位置"""
	notes = []
	for i, code in enumerate(mel):
		if code > 0:
			notes.append((i, code))
	return notes


def get_interval_vector(pcs: List[int]) -> List[int]:
	"""计算音类集合的距离向量 (Interval Vector)"""
	vector = [0] * 6
	unique_pcs = sorted(list(set(pcs)))
	n = len(unique_pcs)
	for i in range(n):
		for j in range(i + 1, n):
			diff = abs(unique_pcs[i] - unique_pcs[j]) % 12
			if diff > 6: diff = 12 - diff
			if diff > 0: vector[diff - 1] += 1
	return vector


def get_rhythmic_onsets(mel: List[int]) -> List[int]:
	"""将旋律转为触发流"""
	return [1 if x > 0 else 0 for x in mel]


def _is_transform_related(pcs1: List[int], pcs2: List[int]) -> Tuple[bool, str]:
	"""判断两个片段是否存在 T, I, R, RI 关系"""
	diffs, sums, rhythm_match = [], [], True
	for p1, p2 in zip(pcs1, pcs2):
		if (p1 > 0 and p2 <= 0) or (p1 <= 0 and p2 > 0) or (p1 <= 0 and p1 != p2):
			rhythm_match = False;
			break
		if p1 > 0 and p2 > 0:
			diffs.append((p2 - p1) % 12)
			sums.append((p1 + p2) % 12)
	if rhythm_match:
		if diffs and len(set(diffs)) == 1: return True, "T"
		if sums and len(set(sums)) == 1: return True, "I"
	if pcs1 == pcs2[::-1]: return True, "R"
	pcs2_rev = pcs2[::-1]
	ri_sums, ri_match = [], True
	for p1, p2r in zip(pcs1, pcs2_rev):
		if (p1 > 0 and p2r <= 0) or (p1 <= 0 and p2r > 0) or (p1 <= 0 and p1 != p2r):
			ri_match = False;
			break
		if p1 > 0 and p2r > 0: ri_sums.append((p1 + p2r) % 12)
	if ri_match and ri_sums and len(set(ri_sums)) == 1: return True, "RI"
	return False, ""


# ================= 3. 评价维度子函数 =================

def _eval_temperament_dimension(real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度一：基于物理律学和音程协和度的评价
	"""
	if len(real_notes) < 2:
		return 1.0  # 单个音符无音程，默认满分
	total_interval_score = 0.0
	count = 0

	for i in range(len(real_notes) - 1):
		n1 = real_notes[i][1]
		n2 = real_notes[i + 1][1]
		semitone_diff = abs(n1 - n2)

		# 1. 基础协和度评分
		# 处理超过八度的音程 (将其折叠到八度内，如 13 -> 1)
		base_diff = semitone_diff % 12
		octave_diff = semitone_diff // 12
		interval_score = INTERVAL_PURITY_SCORE.get(base_diff, 0.0)

		# 2. 惩罚过大的跳进 (课件中提到的不自然音程音)
		# 超过一个八度（12个半音）的跳进在单声部旋律中通常减分
		if semitone_diff > 12:
			interval_score *= 0.5

		# 3. 引入律学偏差补偿
		# 对十二平均律偏离简单整数比的情况进行微调
		deviation = ET_DEVIATION_PENALTY.get(base_diff, 0.0)
		interval_score -= deviation

		total_interval_score += max(0.0, interval_score)
		count += 1

	# 计算平均分
	avg_score = total_interval_score / count if count > 0 else 0.0

	# 4. "狼五度"特别惩罚
	# 上课中提到的：如果在旋律中频繁出现极不稳定的三全音(6)或大七度(11)，而没有得到解决，整体评分会受到非线性惩罚
	dissonance_ratio = sum(1 for n1, n2 in zip(real_notes[:-1], real_notes[1:])
						   if abs(n1[1] - n2[1]) % 12 == 6) / len(real_notes)
	if dissonance_ratio > 0.2:  # 如果超过20%的音程是三全音
		avg_score *= 0.8

	return round(avg_score, 4)


def _eval_tonality_dimension(mel: List[int], real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度二：调性
	1. 严厉惩罚调外音
	2. 奖励强拍（Beat 1, 3）落在大三和弦骨干音（C, E, G）上。
	3. 强化主音（C）在结尾的归属感。
	"""
	if not real_notes: return 0.0
	pcs = [get_pitch_class(n[1]) for n in real_notes]

	# 1. 自然音纯净度
	out_of_scale_count = sum(1 for pc in pcs if pc not in TARGET_SCALE_PCS)
	scale_purity = max(0.0, 1.0 - (out_of_scale_count * 0.2))  # 5个调外音就归零

	# 2. 强拍骨干音
	# 古典旋律中，强拍（0, 4, 8...）应主要由 {0, 4, 7} 构成
	pillar_tones = {0, 4, 7}
	beat_score = 0.0
	beats_checked = 0
	for i in range(0, len(mel), 4):  # 每半个小节检查一次
		beats_checked += 1
		note = mel[i]
		if note > 0:
			pc = get_pitch_class(note)
			if pc in pillar_tones:
				beat_score += 1.0
			elif pc in TARGET_SCALE_PCS:
				beat_score += 0.5
	metrical_stability = beat_score / beats_checked

	# 3. 终止式
	# 最后一小节的最后两个实际音符如果是 G -> C (属->主)，给予较高奖励
	cadence_bonus = 0.0
	if len(pcs) >= 2:
		if pcs[-2] == 7 and pcs[-1] == 0:
			cadence_bonus = 0.3
		elif pcs[-1] == 0:
			cadence_bonus = 0.1

	total_tonality = scale_purity * 0.5 + metrical_stability * 0.3 + cadence_bonus
	return round(min(1.0, total_tonality), 4)


def _eval_pc_set_dimension(real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度三：古典集合色彩
	实测发现奖励用满全音程会与维度二冲突，不再奖励全音程，而是奖励滑动窗口内的音符能否构成大/小三和弦。
	"""
	if len(real_notes) < 3: return 0.5
	pcs = [get_pitch_class(n[1]) for n in real_notes]

	# 目标：大三和弦 [0,0,1,1,1,0] 或 小三和弦 [0,0,1,1,1,0]
	triad_vector = [0, 0, 1, 1, 1, 0]

	match_sum = 0.0
	window_size = 3
	for i in range(len(pcs) - window_size + 1):
		v = get_interval_vector(pcs[i: i + window_size])
		# 计算与三和弦向量的相似度
		diff = sum(abs(v[k] - triad_vector[k]) for k in range(6))
		if diff == 0:
			match_sum += 1.0  # 完美符合三和弦音程关系
		elif diff <= 2:
			match_sum += 0.5

	return round(match_sum / (len(pcs) - window_size + 1), 4)


def _eval_rhythm_dimension(mel: List[int]) -> float:
	"""
	维度四：节奏奇性与活力
	1. 节奏奇性：检测“半周期对称性”。
    2. 轮廓同构：检测节奏/旋律形状的局部同构。
    3. 影子节奏：分析音符触发与节拍重心的重合度。
	4. 约束：惩罚连续休止符。
	"""
	onsets = get_rhythmic_onsets(mel)
	if sum(onsets) < 4: return 0.1  # 音符太少，没有节奏感

	# 1. 节奏密度
	density = sum(onsets) / len(mel)
	density_score = 1.0 - abs(density - IDEAL_DENSITY) * 2

	# 动态推算当前旋律的小节数
	current_num_bars = len(mel) // 8

	# 2. 节奏奇性
	# 根据上课的定义：如果一个节奏型旋转后无法与自身“对角”重合，则具有奇性。
	# 如果强拍和次强拍完全一样（过于对称），奇性分降低。
	oddity_sum = 0.0
	for b in range(current_num_bars):
		bar = onsets[b * 8: (b + 1) * 8]
		# 检查半周期对称性：如果 bar[i] == bar[i+4]，则不具有奇性
		symmetry_points = sum(1 for i in range(4) if bar[i] == bar[i + 4])
		# 对称点越少，奇性越高
		oddity_sum += (1.0 - symmetry_points / 4.0)

	oddity_score = oddity_sum / current_num_bars if current_num_bars > 0 else 0.0

	# 3. 轮廓同构
	# 我们希望节奏有“结构”，即第一小节的节奏型和第三小节最好相似或相同
	# 但如果完全四个小节都一样，则会被惩罚。
	bar_patterns = []
	for b in range(current_num_bars):  # 修改：NUM_BARS -> current_num_bars
		bar_patterns.append(tuple(onsets[b * 8: (b + 1) * 8]))

	unique_patterns = len(set(bar_patterns))
	if unique_patterns == 1:
		isomorphism_score = 0.2  # 极度单调
	elif unique_patterns == 2:
		isomorphism_score = 1.0  # 完美的结构化（如 A-B-A-B）
	elif unique_patterns == 3:
		isomorphism_score = 0.8  # 有变化的结构
	else:
		isomorphism_score = 0.5  # 过于凌乱

	# 4. 影子节奏与稳定性
	# 规则：禁止连续休止符，奖励在弱拍位置的适度触发（影子感）
	shadow_score = 1.0
	consecutive_rests = 0
	for i in range(len(mel) - 1):
		if mel[i] == REST_CODE and mel[i + 1] == REST_CODE:
			consecutive_rests += 1
	shadow_score -= (consecutive_rests * 0.1)

	# 奖励重音位置：检查METRICAL_WEIGHTS的匹配度
	metrical_alignment = sum(onsets[i] * METRICAL_WEIGHTS[i % 8] for i in range(len(onsets)))
	metrical_score = min(1.0, metrical_alignment / (sum(onsets) * 1.0))

	rhythm_total = (
			density_score * 0.2 +
			oddity_score * 0.3 +
			isomorphism_score * 0.3 +
			metrical_score * 0.2
	)

	return round(max(0.0, rhythm_total), 4)


def _eval_symmetry_structure_classical(mel: List[int], real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度五：旋律对称与结构评价
	考虑第一小节（母题）与后续小节的关系。
	   - 如果是四类给定变换关系且不等于原母题：获得高分奖励。
	   - 如果是完全一样的重复：进行惩罚，因为这在进化初期会导致单调。
	结合巴比特的音类覆盖率，奖励在全长内尽可能用满 7 个音类的序列。
	"""
	num_bars = len(mel) // 8
	if num_bars < 1: return 0.5

	bars_pcs = []
	for i in range(num_bars):
		bar_data = mel[i * 8: (i + 1) * 8]
		bars_pcs.append([get_pitch_class(x) if x > 0 else x for x in bar_data])

	# 母题发展
	meaningful_logic = 0
	if num_bars > 1:
		motif = bars_pcs[0]
		for i in range(1, num_bars):
			# 检查与第一小节的对称/移调关系
			is_related, _ = _is_transform_related(motif, bars_pcs[i])
			if is_related and bars_pcs[i] != motif:
				meaningful_logic += 1
		motif_score = min(1.0, (meaningful_logic / (num_bars - 1)) * 1.5) if num_bars > 1 else 1.0
	else:
		motif_score = 0.5  # 单小节无对比，给中值

	# 自然音饱和度 (7音利用率)
	unique_pcs = {get_pitch_class(n[1]) for n in real_notes if n[1] > 0}
	natural_found = unique_pcs.intersection(TARGET_SCALE_PCS)
	# 小节越多，越应该用满 7 个音
	saturation_target = 7 if num_bars >= 4 else (3 + num_bars)
	saturation_score = min(1.0, len(natural_found) / saturation_target)

	# 机械重复与调外音惩罚
	repeats = sum(1 for i in range(1, num_bars) if bars_pcs[i] == bars_pcs[0])
	repeat_penalty = max(0, (repeats - (num_bars // 4)) * 0.15)
	accidental_penalty = sum(0.15 for p in unique_pcs if p not in TARGET_SCALE_PCS)

	final_score = (motif_score * 0.4 + saturation_score * 0.6) - repeat_penalty - accidental_penalty
	return round(max(0.0, min(1.0, final_score)), 4)


# ================= 4. 核心适应度函数 =================

def fitness_enhanced(mel: List[int], num_bars: int = 4) -> float:
	"""
		增强型适应度函数：基于五个维度评价旋律质量
		使用线性插值思想，让权重随小节数平滑过渡。
		接口保持与原项目一致，返回 0.0 ~ 1.0 之间的浮点数。
	"""
	if not mel or all(x <= 0 for x in mel): return 0.0
	real_notes = get_real_notes_with_indices(mel)
	if not real_notes: return 0.0

	# 动态权重分配逻辑
	# 节越多，Symmetry 权重越高 (0.0 -> 0.25)；小节越少，Rhythm 权重越高 (0.25 -> 0.05)
	# 调性和律学作为基础，保持相对稳定
	ratio = min(1.0, (num_bars - 1) / 15.0)  # 1-16小节的进度条

	w_sym = 0.05 + (0.20 * ratio)  # 5% 到 25%
	w_rhy = 0.20 - (0.15 * ratio)  # 20% 到 5%
	w_tonal = 0.45 - (0.10 * ratio)  # 45% 到 35% (长旋律调性略微放宽给结构腾位置)
	w_temp = 0.30  # 恒定 30%
	w_pc = 1.0 - (w_sym + w_rhy + w_tonal + w_temp)  # 剩余给局部色彩

	# 作为参考，8 小节时的参数如下
	# "temperament": 0.3000  # 律学与音程协和度
	# "tonality": 0.4033     # 调式功能
	# "pc_set": 0.0234       # 音类集合色彩
	# "rhythm": 0.1300       # 节奏奇性与活力
	# "symmetry": 0.1433     # 旋律对称与结构冗余

	# 调用子函数
	score_temp = _eval_temperament_dimension(real_notes)
	score_tonal = _eval_tonality_dimension(mel, real_notes)
	score_pcset = _eval_pc_set_dimension(real_notes)
	score_rhythm = _eval_rhythm_dimension(mel)
	score_symmetry = _eval_symmetry_structure_classical(mel, real_notes)

	total_score = (
			score_temp * w_temp +
			score_tonal * w_tonal +
			score_pcset * w_pc +
			score_rhythm * w_rhy +
			score_symmetry * w_sym
	)
	return round(float(total_score), 4)


# ================= 5. 遗传算法主流程 =================

def run(alpha: float = 0.5,
		m=100,
		n=10,
		crossover_probability=0.2,
		mutation_probability=0.5,
		transposition_probability=0.12,
		retrograde_probability=0.06,
		inversion_probability=0.09,
		retrograde_inversion_probability=0.03,
		look_ahead_steps=20,  # 新增：局部搜索步数
		fitness=fitness_enhanced,  # 新增：选择哪个函数作为适应度函数
		adapt_ratio_set=0.9, # 新增：当适应度卡住150步时，大幅上调变异率为该数
		num_bars=4):  # 新增：小节数参数
	"""
	运行优化后的遗传算法（引入局部贪心搜索与自适应变异策略）
	"""

	# 1. 初始化种群
	population = [generate_random_melody(num_bars=num_bars) for _ in range(n)]

	# 用于监控停滞状态的变量
	best_fitness_so_far = -1.0
	stagnation_counter = 0

	# 2. 迭代 m 次
	for gen in range(m):
		fitnesses = [fitness(mel, num_bars=num_bars) for mel in population]

		# 每一百轮打印当前代最高分，方便观察增长过程
		current_max = max(fitnesses)

		# --- 自适应逻辑 —— 更新停滞计数器 ---
		if current_max > best_fitness_so_far:
			best_fitness_so_far = current_max
			stagnation_counter = 0
		else:
			stagnation_counter += 1

		# --- 计算自适应调整因子 (建议0.0 到 0.9 之间) ---
		# 逻辑：如果连续 300 代没有进步，调整因子达到最大(默认0.9)。可以根据需要调整 150 这个阈值。
		adapt_ratio = min(stagnation_counter / 150, adapt_ratio_set)

		# 动态计算三个操作的边界，实现“此消彼长”
		# 1. 缩小交叉概率区间
		dyn_crossover_bound = crossover_probability * (1 - adapt_ratio)

		# 2. 增大变异概率区间，同时挤压“音乐变换”的空间
		# 原始变异结束边界
		orig_mutation_end = crossover_probability + mutation_probability
		# 变异边界向 1.0 靠拢，从而压缩后面的 else (变换) 概率
		dyn_mutation_bound = 1.0 - (1.0 - orig_mutation_end) * (1 - adapt_ratio)
		# 变异实际占用的区间长度为 (dyn_mutation_bound - dyn_crossover_bound)

		if gen % 100 == 0:
			# 打印日志时顺便观察停滞计数和当前的动态边界
			print(f"Generation {gen}: Max Fitness = {current_max}, Stagnation = {stagnation_counter}")

		# 2.3 进化生成下一代
		next_generation = []

		# 为了不让高分个体停滞，我们也对高分个体进行“贪心优化”
		# 先把表现好的选出来
		sorted_indices = np.argsort(fitnesses)[::-1]

		# 保持精英策略：将最顶尖的个体进行贪心尝试后再放入下一代
		elite_count = max(1, n // 10)
		for i in range(elite_count):
			elite = population[sorted_indices[i]]
			# 对精英进行多步“深挖”优化
			for _ in range(look_ahead_steps):
				candidate = mutation(elite)
				if fitness(candidate, num_bars=num_bars) > fitness(elite, num_bars=num_bars):
					elite = candidate
			next_generation.append(elite)

		# 填充剩余位置
		while len(next_generation) < n:
			k = random.random()

			# --- 应用动态概率边界 ---
			if k < dyn_crossover_bound:
				# 交叉操作
				idx1 = roulette_wheel_selection(n, fitnesses)
				idx2 = roulette_wheel_selection(n, fitnesses)
				c1, c2 = crossover(population[idx1], population[idx2])
				next_generation.append(c1)
				if len(next_generation) < n:
					next_generation.append(c2)

			elif k < dyn_mutation_bound:
				# --- 变异优化 (在停滞时此分支概率会显著提升) ---
				selected_idx = roulette_wheel_selection(n, fitnesses)
				current_mel = population[selected_idx].copy()
				current_fit = fitness(current_mel, num_bars=num_bars)

				# 尝试多步变异（向前看 look_ahead_steps 步）
				best_local_mel = current_mel
				best_local_fit = current_fit

				temp_mel = current_mel
				for _ in range(look_ahead_steps):
					# 在当前基础上继续变异（寻找更深处的可能）
					temp_mel = mutation(temp_mel)
					temp_fit = fitness(temp_mel, num_bars=num_bars)

					if temp_fit > best_local_fit:
						best_local_mel = temp_mel
						best_local_fit = temp_fit

				# 如果这几步尝试确实变好了，就用变好后的；否则保留原样（或只取变好后的部分）
				next_generation.append(best_local_mel)

			else:
				# 其他音乐变换（移调、逆行等）
				# 此分支在停滞时的触发概率会被 dyn_mutation_bound 的上移而压缩
				selected_idx = roulette_wheel_selection(n, fitnesses)
				parent = population[selected_idx]

				# 随机选择一种音乐变换并评估
				child = parent.copy()
				transform_choice = random.random()
				transform_sum = (transposition_probability + retrograde_probability +
								 inversion_probability + retrograde_inversion_probability)
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

				# 贪心：如果变换后适应度严重下降，则进行一次变异补偿
				if fitness(child, num_bars=num_bars) < fitness(parent, num_bars=num_bars):
					for _ in range(look_ahead_steps):
						improved_child = mutation(child)
						if fitness(improved_child, num_bars=num_bars) > fitness(child, num_bars=num_bars):
							child = improved_child

				next_generation.append(child)

		population = next_generation[:n]

	# 3. 返回最终结果
	result = sorted(population, key=lambda mel: fitness(mel, num_bars=num_bars), reverse=True)
	# 只返回满足阈值的
	final_filtered = [mel for mel in result if fitness(mel, num_bars=num_bars) >= alpha]

	if not final_filtered:
		print("没有找到达到阈值的旋律，返回当前最优个体")
		return [result[0]]

	print("最终最高适应度：", fitness(final_filtered[0], num_bars=num_bars))
	return final_filtered

if __name__ == "__main__":
	# 支持1-16小节生成。建议n、look_ahead_steps、m随生成小节数上升而适当上升
	# 小节数上升时需要更大的种群提供交叉，需要更深地探索局部性，需要更多的迭代轮次以理解规则
	# 8 小节时，参考：alpha=0.85,n=15,look_ahead_steps=10, m=1500,fitness=fitness_enhanced,num_bars=8
	# 16 小节时，参考：alpha=0.85,n=20,look_ahead_steps=20, m=500(时间妥协),fitness=fitness_enhanced,num_bars=16
	result = run(alpha=0.85,n=20,look_ahead_steps=20, m=500,fitness=fitness_enhanced,num_bars=8)
	print(result)
	for idx, mel in enumerate(result):
		output_path = f"./output/genetic_algorithm_result_{idx + 1}.wav"
		synthesize_melody(
			codes=mel,
			output_path=output_path,
			sample_dir="./samples/",
			BPM=167,
			unit_time=180
		)
