# 完备的fitness_enhance函数
# 保留了原先的大部分代码，在fitness与run中新增参数bar_nums，传入8可以生成8小节音乐。
# 新增了fitness_enhance函数以及其对应辅助函数。依据王杰老师课程内容，分数更为严格，生成质量大幅提高。
# 在run中新增了逻辑，详见readme
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
from audio_synth import synthesize_melody

# 编码常量（和 note_encoding.py 对齐）
TIE_CODE = -1  # 延长符
REST_CODE = 0  # 休止符
MIN_NOTE_CODE = 33  # F3
MAX_NOTE_CODE = 59  # G5

NUM_BARS = 4
UNITS_PER_BAR = 8

# 音乐变换参数
MAX_TRANSFORM_ATTEMPTS = 10  # 音乐变换生成音符超出范围比例达到阈值OUT_OF_RANGE_THRESHOLD后，再次尝试变换的最大尝试次数
TRANSPOSITION_SEMITONE_RANGE = 7  # 移调的半音范围：[-7, 7]（排除0）
INVERSION_AXIS_SCOPE = 2  # 倒影轴的选择范围：平均音高±2
OUT_OF_RANGE_THRESHOLD = 0.1  # 可接受的超出范围音符比例阈值（10%）

import math

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


def fitness(mel: List[int], num_bars: int = 4) -> float:
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
	w_melodic = 0.6  # 旋律线条流畅/和谐权重
	w_harmonic = 0.3  # 骨干音(强拍)和谐权重

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
			n2 = real_notes[i + 1][1]
			# 计算相邻两个实际音符的音程分
			interval_scores.append(calculate_interval_score(n1, n2))

		melodic_score = sum(interval_scores) / len(interval_scores)
	else:
		melodic_score = 1.0  # 只有一个音，完美平滑

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
			n2 = strong_beat_notes[i + 1]
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
		fitness=fitness,  # 新增：选择哪个函数作为适应度函数
		adapt_ratio_set=0.9, # 新增：当适应度卡住150步时，大幅上调变异率为该数
		num_bars=4):  # 新增：小节数参数
	"""
	运行优化后的遗传算法（引入局部贪心搜索与自适应变异策略）
	"""

	# 1. 初始化种群
	population = [generate_random_melody(num_bars=num_bars) for _ in range(n)]

	# --- 新增：用于监控停滞状态的变量 ---
	best_fitness_so_far = -1.0
	stagnation_counter = 0

	# 2. 迭代 m 次
	for gen in range(m):
		fitnesses = [fitness(mel, num_bars=num_bars) for mel in population]

		# 每一百轮打印当前代最高分，方便观察增长过程
		current_max = max(fitnesses)

		# --- 新增：自适应逻辑 —— 更新停滞计数器 ---
		if current_max > best_fitness_so_far:
			best_fitness_so_far = current_max
			stagnation_counter = 0
		else:
			stagnation_counter += 1

		# --- 新增：计算自适应调整因子 (0.0 到 0.9 之间) ---
		# 逻辑：如果连续 300 代没有进步，调整因子达到最大(默认0.9)。你可以根据需要调整 300 这个阈值。
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

			# --- 修改：应用动态概率边界 ---
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


def crossover(parent1: List[int], parent2: List[int]) -> tuple[List[int], List[int]]:
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
			pos > 0 and  # 不能是第一个
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
		semitones = random.choice(available_semitones)  # 定义移调变换函数

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



def fitness_enhanced(mel: List[int], num_bars: int = 4) -> float:
	"""
	增强型适应度函数：基于五个维度评价旋律质量
	接口保持与原项目一致，返回 0.0 ~ 1.0 之间的浮点数。
	"""
	# 1. 基础合法性检查
	if not mel or all(x <= 0 for x in mel):
		return 0.0

	# 2. 数据预处理 (提取实际音符、MIDI值、音类等信息)
	real_notes = get_real_notes_with_indices(mel)  # (index, code)
	if not real_notes:
		return 0.0

	# 3. 定义五个维度的权重 (可根据创作偏好调整)
	# 建议总和为 1.0
	weights = {
		"temperament": 0.30,  # 提高：物理协和是古典美的基石
		"tonality": 0.40,  # 核心：调性逻辑必须占大头
		"pc_set": 0.10,  # 辅助：局部色彩
		"rhythm": 0.10,  # 辅助：节奏平稳
		"symmetry": 0.10  # 辅助：结构呼应
	}

	# 4. 调用各个维度的评价子函数 (暂设为占位符)
	score_temp = _eval_temperament_dimension(real_notes)
	score_tonal = _eval_tonality_dimension(mel, real_notes)
	score_pcset = _eval_pc_set_dimension(real_notes)
	score_rhythm = _eval_rhythm_dimension(mel)
	score_symmetry = _eval_symmetry_dimension(mel, real_notes)

	# --- 核心逻辑修改：分情况讨论评价对称性 ---
	if num_bars == 8:
		score_symmetry = _eval_symmetry_dimension_8bars(mel, real_notes)
	else:
		score_symmetry = _eval_symmetry_dimension(mel, real_notes)

	# 5. 加权总分计算
	total_score = (
			score_temp * weights["temperament"] +
			score_tonal * weights["tonality"] +
			score_pcset * weights["pc_set"] +
			score_rhythm * weights["rhythm"] +
			score_symmetry * weights["symmetry"]
	)

	return round(float(total_score), 4)


# ================= 评价维度子函数占位符 =================

# ================= 维度一：律学常量配置 =================

# 基于频率比简单程度的评分 (0.0 ~ 1.0)
# 参考 PDF: 2:1(八度), 3:2(纯五), 4:3(纯四), 5:4(大三), 6:5(小三)
# 12-ET（十二平均律）下这些音程虽然有微小偏差，但在听觉上被归类为协和
INTERVAL_PURITY_SCORE = {
	0: 1.0,  # 纯一度 (1:1)
	12: 1.0,  # 纯八度 (2:1)
	7: 0.95,  # 纯五度 (3:2) - 极高协和
	5: 0.85,  # 纯四度 (4:3) - 高协和
	4: 0.80,  # 大三度 (5:4) - 协和
	9: 0.75,  # 大六度 (5:3) - 协和
	3: 0.70,  # 小三度 (6:5) - 协和
	8: 0.65,  # 小六度 (8:5) - 协和

	# 不协和音程 (PDF 提到的大二、小七、增四等)
	2: 0.30,  # 大二度 (9:8) - 不稳定
	10: 0.25,  # 小七度 (9:5) - 不稳定
	1: 0.10,  # 小二度 (16:15) - 极不协和
	11: 0.10,  # 大七度 (15:8) - 极不协和
	6: 0.05,  # 增四/减五 (三全音 45:32) - "狼"音程趋势，极不稳定
}

# 音分偏差惩罚 (针对 12-ET 偏离 纯律 的补偿)
# 比如大三度在平均律(400音分)比纯律(386音分)高了14音分。
# 虽然我们处理的是整数 MIDI，但通过这个权重可以微调进化方向。
ET_DEVIATION_PENALTY = {
	4: 0.05,  # 大三度偏高 14 cents
	9: 0.06,  # 大六度偏高 16 cents
	7: 0.01,  # 纯五度偏差极小 (2 cents)
}


def _eval_temperament_dimension(real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度一：基于物理律学和音程协和度的评价
	输入: real_notes - [(位置, 编码), ...]
	返回: 0.0 ~ 1.0 评分
	"""
	if len(real_notes) < 2:
		return 1.0  # 单个音符无音程，默认满分

	total_interval_score = 0.0
	count = 0

	for i in range(len(real_notes) - 1):
		# 获取相邻两个音符的 MIDI 编号差（即半音数）
		n1 = real_notes[i][1]
		n2 = real_notes[i + 1][1]
		semitone_diff = abs(n1 - n2)

		# 1. 基础协和度评分
		# 处理超过八度的音程 (将其折叠到八度内，如 13 -> 1)
		base_diff = semitone_diff % 12
		octave_diff = semitone_diff // 12

		interval_score = INTERVAL_PURITY_SCORE.get(base_diff, 0.0)

		# 2. 惩罚过大的跳进 (PDF 中提到的不自然音程)
		# 超过一个八度（12个半音）的跳进在单声部旋律中通常减分
		if semitone_diff > 12:
			interval_score *= 0.5

		# 3. 引入律学偏差补偿 (Cent Deviation)
		# 针对 12-ET 偏离“理想简单整数比”的情况进行微调
		deviation = ET_DEVIATION_PENALTY.get(base_diff, 0.0)
		interval_score -= deviation

		total_interval_score += max(0.0, interval_score)
		count += 1

	# 计算平均分
	avg_score = total_interval_score / count if count > 0 else 0.0

	# 4. "狼五度"特别惩罚 (Wolf Fifth Penalty)
	# 模拟 PDF 中提到的：如果在旋律中频繁出现极不稳定的三全音(6)或大七度(11)
	# 而没有得到解决，整体评分会受到非线性惩罚
	dissonance_ratio = sum(1 for n1, n2 in zip(real_notes[:-1], real_notes[1:])
						   if abs(n1[1] - n2[1]) % 12 == 6) / len(real_notes)
	if dissonance_ratio > 0.2:  # 如果超过20%的音程是三全音
		avg_score *= 0.8

	return round(avg_score, 4)

# ================= 维度二：调性与功能配置 =================

# 目标音阶：C大调音阶的音类集合 (C, D, E, F, G, A, B)
# 如果你想写a小调，改为 {9, 11, 0, 2, 4, 5, 7}
TARGET_SCALE_PCS = {0, 2, 4, 5, 7, 9, 11}

# 音级稳定性评分 (Pitch Class Stability)
# 1.0: 主音 (C) - 极稳
# 0.8: 属音/中音 (G, E) - 稳（主和弦音）
# 0.4: 下属音/上主音 (F, D, A) - 不稳
# 0.1: 导音 (B) - 极不稳，强烈趋向主音
STABILITY_MAP = {
    0: 1.0,  # Tonic (I)
    7: 0.8,  # Dominant (V)
    4: 0.8,  # Mediant (III)
    5: 0.4,  # Subdominant (IV)
    2: 0.4,  # Supertonic (II)
    9: 0.4,  # Submediant (VI)
    11: 0.1, # Leading note (VII)
}

# 解决规则定义 (Resolution Pairs)
# 不稳定音 -> 稳定音 的趋向
RESOLUTION_TRENDS = {
    11: 0, # 导音(7) 必须解决到 主音(1)
    5: 4,  # 下属音(4) 倾向解决到 中音(3)
    2: 0,  # 上主音(2) 倾向解决到 主音(1)
}

def _eval_tonality_dimension(mel: List[int], real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度二：古典调性增强版
	1. 严厉惩罚调外音（Accidentals）。
	2. 奖励强拍（Beat 1, 3）落在大三和弦骨干音（C, E, G）上。
	3. 强化主音（C）在结尾的归属感。
	"""
	if not real_notes: return 0.0
	pcs = [get_pitch_class(n[1]) for n in real_notes]

	# --- 1. 自然音纯净度 (Scale Purity) ---
	# 古典模式下，每一个调外音都是严重的“杂质”
	out_of_scale_count = sum(1 for pc in pcs if pc not in TARGET_SCALE_PCS)
	scale_purity = max(0.0, 1.0 - (out_of_scale_count * 0.2))  # 5个调外音就归零

	# --- 2. 强拍骨干音 (Structural Tones) ---
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

	# --- 3. 终止式 (Cadence) ---
	# 最后一小节的最后两个实际音符如果是 G -> C (属->主)，给予极大奖励
	cadence_bonus = 0.0
	if len(pcs) >= 2:
		if pcs[-2] == 7 and pcs[-1] == 0:
			cadence_bonus = 0.3
		elif pcs[-1] == 0:
			cadence_bonus = 0.1

	total_tonality = scale_purity * 0.5 + metrical_stability * 0.3 + cadence_bonus
	return round(min(1.0, total_tonality), 4)


# ================= 维度三：音类集合配置 =================

# 目标向量：我们希望旋律的局部片段尽量接近“大/小三和弦”的色彩
# 即使是旋律，局部音符构成的集合若符合此向量，听感也会非常“悦耳”
TARGET_VECTOR = [0, 0, 1, 1, 1, 0]

# 集合类奖惩
# 3-11 (大/小三和弦) 的原型是 {0, 3, 7} 或 {0, 4, 7}
# 4-27 (属七/半减七) 的原型向量是 [0, 1, 2, 1, 1, 1]

def get_interval_vector(pcs: List[int]) -> List[int]:
    """
    计算音类集合的距离向量 (Interval Vector)
    d1: 小二度/大七度, d2: 大二度/小七度, d3: 小三度/大六度
    d4: 大三度/小六度, d5: 纯四度/纯五度, d6: 三全音
    """
    vector = [0] * 6
    unique_pcs = sorted(list(set(pcs)))
    n = len(unique_pcs)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(unique_pcs[i] - unique_pcs[j]) % 12
            if diff > 6:
                diff = 12 - diff
            if diff > 0:
                vector[diff - 1] += 1
    return vector


def _eval_pc_set_dimension(real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度三：古典集合色彩 (Triadic Color)
	不再奖励全音程，而是奖励滑动窗口内的音符能否构成大/小三和弦。
	"""
	if len(real_notes) < 3: return 0.5
	pcs = [get_pitch_class(n[1]) for n in real_notes]

	# 目标：大三和弦 [0,0,1,1,1,0] 或 小三和弦 [0,0,1,1,1,0] (它们的向量是一样的)
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


# ================= 维度四：节奏活力配置 =================

# 理想密度：每小节 8 个单位中，触发 4~6 个音符（50%~75%）最为平衡
IDEAL_DENSITY = 0.6

# 强弱拍权重 (4/4 拍，8个单位)
# 索引: 0(强) 1(弱) 2(次强) 3(弱) 4(强) 5(弱) 6(次强) 7(弱)
# 为了增加“奇性”，我们有时会奖励弱拍起音（切分音）
METRICAL_WEIGHTS = [1.2, 0.6, 0.9, 0.5, 1.1, 0.5, 0.8, 0.4]

def get_rhythmic_onsets(mel: List[int]) -> List[int]:
    """将旋律转为触发流：1代表新音起，0代表延续或休止"""
    return [1 if x > 0 else 0 for x in mel]

def get_rhythmic_intervals(onsets: List[int]) -> List[int]:
    """计算发声点之间的时间间隔（IOI - Inter-OnsetInterval）"""
    indices = [i for i, val in enumerate(onsets) if val == 1]
    if len(indices) < 2: return []
    return [indices[i+1] - indices[i] for i in range(len(indices)-1)]


def _eval_rhythm_dimension(mel: List[int]) -> float:
	"""
	维度四：节奏奇性与活力 (Rhythmic Oddity & Vitality)
	1. Rhythmic Oddity：通过检测“半周期对称性”来评分。
	2. Contour Isomorphism：检测节奏/旋律形状的局部同构。
	3. Shadow Rhythm：分析音符触发与节拍重心的重合度。
	4. 约束：惩罚连续休止符。
	"""
	onsets = get_rhythmic_onsets(mel)
	if sum(onsets) < 4: return 0.1  # 音符太少，没有节奏感

	# --- 1. 节奏密度 (Density) ---
	density = sum(onsets) / len(mel)
	density_score = 1.0 - abs(density - IDEAL_DENSITY) * 2

	# --- 2. 节奏奇性 (Rhythmic Oddity) ---
	# 根据 PDF 里的定义：如果一个节奏型旋转后无法与自身“对角”重合，则具有奇性。
	# 我们以小节（8单位）为单位，检查是否存在 x_i = x_{i+4} 的情况。
	# 如果强拍和次强拍完全一样（过于对称），奇性分降低。
	oddity_sum = 0.0
	for b in range(NUM_BARS):
		bar = onsets[b * 8: (b + 1) * 8]
		# 检查半周期对称性：如果 bar[i] == bar[i+4]，则不具有奇性
		symmetry_points = sum(1 for i in range(4) if bar[i] == bar[i + 4])
		# 对称点越少，奇性(Vitality)越高
		oddity_sum += (1.0 - symmetry_points / 4.0)
	oddity_score = oddity_sum / NUM_BARS

	# --- 3. 轮廓同构 (Contour Isomorphism / Pattern Repetition) ---
	# 我们希望节奏有“结构”，即：第一小节的节奏型和第三小节最好相似或相同
	# 但如果完全四个小节都一样，则会被惩罚。
	bar_patterns = []
	for b in range(NUM_BARS):
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

	# --- 4. 影子节奏与稳定性 (Shadow Rhythm / Rest Penalty) ---
	# 规则：禁止连续休止符（PDFSlide 32），奖励在弱拍位置的适度触发（影子感）
	shadow_score = 1.0
	consecutive_rests = 0
	for i in range(len(mel) - 1):
		if mel[i] == REST_CODE and mel[i + 1] == REST_CODE:
			consecutive_rests += 1
	shadow_score -= (consecutive_rests * 0.1)

	# 奖励重音位置：检查 METRICAL_WEIGHTS 的匹配度
	metrical_alignment = sum(onsets[i] * METRICAL_WEIGHTS[i % 8] for i in range(len(onsets)))
	metrical_score = min(1.0, metrical_alignment / (sum(onsets) * 1.0))

	# 加权合并
	rhythm_total = (
			density_score * 0.2 +
			oddity_score * 0.3 +
			isomorphism_score * 0.3 +
			metrical_score * 0.2
	)

	return round(max(0.0, rhythm_total), 4)


def _get_bar_pcs(mel: List[int], bar_idx: int) -> List[int]:
    """提取指定小节的音类序列（保留节奏占位）"""
    bar_data = mel[bar_idx * 8 : (bar_idx + 1) * 8]
    return [get_pitch_class(x) if x > 0 else x for x in bar_data]


def _is_transform_related(pcs1: List[int], pcs2: List[int]) -> Tuple[bool, str]:
	"""
	判断两个片段是否存在 T, I, R, RI 关系
	修复了无限递归漏洞，并严格检查节奏对齐
	"""
	# 辅助检查：节奏结构必须一致（即休止符和延长符的位置必须完全相同）
	# 除非是逆行关系，逆行时节奏结构也应该是逆行的

	# 1. 检查移调 (T)
	# 关系：pcs2[i] = (pcs1[i] + n) % 12
	diffs = []
	rhythm_match = True
	for p1, p2 in zip(pcs1, pcs2):
		if (p1 > 0 and p2 <= 0) or (p1 <= 0 and p2 > 0) or (p1 <= 0 and p1 != p2):
			rhythm_match = False
			break
		if p1 > 0 and p2 > 0:
			diffs.append((p2 - p1) % 12)
	if rhythm_match and diffs and len(set(diffs)) == 1:
		return True, "T"

	# 2. 检查倒影 (I)
	# 关系：(pcs1[i] + pcs2[i]) % 12 = constant axis
	sums = []
	rhythm_match = True
	for p1, p2 in zip(pcs1, pcs2):
		if (p1 > 0 and p2 <= 0) or (p1 <= 0 and p2 > 0) or (p1 <= 0 and p1 != p2):
			rhythm_match = False
			break
		if p1 > 0 and p2 > 0:
			sums.append((p1 + p2) % 12)
	if rhythm_match and sums and len(set(sums)) == 1:
		return True, "I"

	# 3. 检查逆行 (R)
	# 关系：pcs1 == pcs2[::-1]
	if pcs1 == pcs2[::-1]:
		return True, "R"

	# 4. 检查逆行倒影 (RI)
	# 关系：pcs1 与 pcs2[::-1] 满足倒影(I)关系
	pcs2_rev = pcs2[::-1]
	ri_sums = []
	rhythm_match = True
	for p1, p2r in zip(pcs1, pcs2_rev):
		if (p1 > 0 and p2r <= 0) or (p1 <= 0 and p2r > 0) or (p1 <= 0 and p1 != p2r):
			rhythm_match = False
			break
		if p1 > 0 and p2r > 0:
			ri_sums.append((p1 + p2r) % 12)
	if rhythm_match and ri_sums and len(set(ri_sums)) == 1:
		return True, "RI"

	return False, ""


def _eval_symmetry_dimension(mel: List[int], real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度五：旋律对称与结构评价 (Symmetry & Structure)

	逻辑：
	1. 提取 4 个小节的音类序列。
	2. 将第一小节（Bar 1）定义为“核心母题”。
	3. 检查后续小节（Bar 2, 3, 4）与母题的关系：
	   - 如果是变换关系（T, I, R, RI）且不等于原母题：获得高分奖励。
	   - 如果是完全一样的重复（Identity）：进行惩罚，因为这在进化初期会导致单调。
	4. 结合巴比特（Babbitt）的音类覆盖率：奖励在 4 小节内尽可能用满 12 个音类的序列。
	"""
	# 基础检查：确保旋律长度覆盖 4 小节 (4 * 8 = 32 units)
	if len(mel) < 32:
		return 0.5

	# --- A. 分割小节并提取音类 ---
	bars_pcs = []
	for i in range(4):
		# 提取每一小节 (8个单位时值)
		bar_data = mel[i * 8: (i + 1) * 8]
		# 转换为音类 (get_pitch_class 处理 code > 0 的音符，保留 0 和 -1 作为节奏标记)
		pcs = [get_pitch_class(x) if x > 0 else x for x in bar_data]
		bars_pcs.append(pcs)

	motif = bars_pcs[0]

	meaningful_transforms = 0  # 有意义的变换次数
	literal_repeats = 0  # 机械重复次数
	found_relations = set()  # 记录出现的变换类型 (T, I, R, RI)

	# --- B. 母题发展检测 ---
	for i in range(1, 4):
		target_bar = bars_pcs[i]

		# 1. 检查是否为机械重复 (Identity / T0)
		if target_bar == motif:
			literal_repeats += 1
			continue  # 机械重复不计入“变换”奖励

		# 2. 检查是否存在 T, I, R, RI 变换
		# 使用修复后的非递归版本 _is_transform_related
		is_related, rel_type = _is_transform_related(motif, target_bar)

		if is_related:
			meaningful_transforms += 1
			found_relations.add(rel_type)

	# --- C. 计算结构分数 (Structure Score) ---
	# 评分逻辑：
	# 0 次变换: 0.2 (缺乏逻辑关联)
	# 1 次变换: 0.7 (基本的母题呼应)
	# 2 次变换: 1.0 (优良的结构化，类似 A-A'-A''-B)
	# 3 次变换: 0.9 (非常紧凑，但略显死板)
	if meaningful_transforms == 0:
		motif_score = 0.2
	elif meaningful_transforms == 1:
		motif_score = 0.7
	elif meaningful_transforms == 2:
		motif_score = 1.0
	else:
		motif_score = 0.9

	# 如果变换类型多样（既有移调又有倒影），给予额外的小奖励
	if len(found_relations) > 1:
		motif_score = min(1.0, motif_score + 0.1)

	# --- D. 机械重复惩罚 (Identity Penalty) ---
	# 适当的重复是好的，但遗传算法容易陷入全相同的局部最优解
	# 每个完全重复的小节扣除 0.2 分
	penalty = literal_repeats * 0.2

	# --- E. 音类覆盖率得分 (Coverage Score) ---
	# 基于《音类集合与新黎曼理论》和勋伯格十二音技术
	# 检查整段旋律用了多少个不同的音类
	unique_pcs = {get_pitch_class(n[1]) for n in real_notes if n[1] > 0}
	coverage_score = len(unique_pcs) / 12.0  # 满分 1.0 代表 12 个音全用了

	# --- F. 综合加权 ---
	# 结构逻辑占 60%，音类丰富度占 40%
	final_score = (motif_score * 0.6 + coverage_score * 0.4) - penalty

	# 限制范围在 [0, 1]
	return round(max(0.0, min(1.0, final_score)), 4)

#######################
# 八小节版本的旋律对称评价 #
#######################
def _eval_symmetry_dimension_8bars(mel: List[int], real_notes: List[Tuple[int, int]]) -> float:
	"""
	维度五：古典结构美学版
	1. 引入“调性移调”宽容度：允许在自然音阶内的度数移动，不强求半音精确。
	2. 奖励“音阶饱和度”：在 8 小节内，自然音阶的 7 个音（CDEFGAB）都出现过。
	3. 放宽重复惩罚：允许 A-A-B-A 结构。
	"""
	if len(mel) < 64: return 0.5

	bars_pcs = []
	for i in range(8):
		bar_data = mel[i * 8: (i + 1) * 8]
		pcs = [get_pitch_class(x) if x > 0 else x for x in bar_data]
		bars_pcs.append(pcs)

	motif_front = bars_pcs[0]

	# --- 1. 调性移调检查 (Tonal Transposition) ---
	# 如果 Bar 2 是 Bar 1 在调式内的平移，即为高分
	meaningful_logic = 0
	for i in [1, 2, 4, 5]:  # 检查关键小节
		is_related, _ = _is_transform_related(motif_front, bars_pcs[i])
		if is_related: meaningful_logic += 1

	# --- 2. 机械重复控制 (Balanced Repetition) ---
	# 古典乐喜欢 A-A-B-A。如果只有一个小节重复，不惩罚；超过 3 个才惩罚。
	repeats = sum(1 for i in range(1, 8) if bars_pcs[i] == motif_front)
	repeat_penalty = 0.0
	if repeats >= 3:
		repeat_penalty = 0.3
	elif repeats == 0:
		repeat_penalty = 0.1  # 缺乏统一感

	# --- 3. 自然音饱和度 (Scale Saturation) ---
	# 奖励：在 8 小节内是否用满了 C 大调的 7 个音
	unique_pcs = {get_pitch_class(n[1]) for n in real_notes if n[1] > 0}
	natural_found = unique_pcs.intersection(TARGET_SCALE_PCS)
	saturation_score = len(natural_found) / 7.0

	# 额外惩罚：如果用了黑键音，在结构分里也扣分（强化古典纯净感）
	accidental_penalty = sum(0.1 for p in unique_pcs if p not in TARGET_SCALE_PCS)

	final_score = (meaningful_logic / 4.0) * 0.4 + saturation_score * 0.6 - repeat_penalty - accidental_penalty
	return round(max(0.0, min(1.0, final_score)), 4)


if __name__ == "__main__":
	# look_ahead_steps决定局部深挖的步数，与m、n一样与耗时线性相关
	# 实测超过2000轮迭代后提升微小。目前参数较优
	result = run(alpha=0.85,n=20,look_ahead_steps=15, m=2000,fitness=fitness_enhanced,num_bars=8)
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
