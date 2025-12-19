"""
llm 适应度模块
"""

import random
import numpy as np
from typing import List, Tuple
from population import generate_random_melody
from util.audio_synth import synthesize_melody
from util.note_encoding import _midi_to_note_name
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor 
import os
from config import *
from genetics import (MAX_TRANSFORM_ATTEMPTS, TRANSPOSITION_SEMITONE_RANGE, INVERSION_AXIS_SCOPE, OUT_OF_RANGE_THRESHOLD,
                      roulette_wheel_selection, crossover, mutation, _apply_and_check_transform, _is_transform_acceptable,
                      transposition, retrograde, inversion, retrograde_inversion)


client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'), 
    base_url="https://api.deepseek.com"
)

# 全局缓存，避免重复评估同一段旋律
fitness_cache = {}

def fitness(mel: List[int]) -> float:
    """
    使用 DeepSeek LLM 评估旋律的适应度。
    范围 0.0 - 1.0
    """
    # 1. 转换 list 为 tuple 以便作为字典 key (缓存查找)
    mel_tuple = tuple(mel)
    if mel_tuple in fitness_cache:
        return fitness_cache[mel_tuple]

    # 2. 构造 Prompt
    # 将数组转换为字符串形式，方便 LLM 阅读
    tokens: List[str] = []

    for code in mel:
        code = int(code)
        if code == TIE_CODE:
            tokens.append("-")
        elif code == REST_CODE:
            tokens.append("0")
        else:
            if not (1 <= code <= 88):
                raise ValueError(
                    f"编码 {code} 不在合法音高编码范围 1..88 或特例 -1,0 内"
                )
            midi = code + 20  # 1->21, 88->108
            note_name = _midi_to_note_name(midi)
            tokens.append(note_name)
    
    melody_str = " ".join(tokens)
    print(melody_str)
    
    system_prompt = """
You are a professional music composer and music theory expert. 
Your task is to evaluate a short melody sequence based on its musicality, structure, and pleasantness.
    """

    user_prompt = f"""
Please evaluate the following melody sequence and provide a fitness score from 0.0 to 1.0.

### Melody Constraints & Encoding:
- **Time Signature**: 4/4
- **Length**: 4 bars, 32 units total (8 units per bar).
- "0": Rest
- "-": Enumeration symbol
- C5, #F4, bG3, etc.: Note names

### Evaluation Criteria:
1. **Musicality**: Does the melody have a good contour? Is it catchy?
2. **Rhythm**: Are the notes and rests distributed logically? (Avoid chaotic syncopation unless musical).
3. **Voice Leading**: Are intervals smooth? (Avoid random large jumps).
4. **Structure**: Does it feel like it has a beginning and an end?
5. **Validity**: It should utilize the range F3-G5 effectively.

### The Melody Data:
{melody_str}

### Output Requirement:
Return **ONLY** a single floating-point number between 0.0 and 1.0. Do not write any explanation, text, or markdown. Just the number.
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=1
        )
        
        content = response.choices[0].message.content.strip()
        
        # 清理可能出现的额外字符 (比如 LLM 输出了 "Score: 0.8")
        import re
        # 提取数字 (支持 0.9, .9, 0.95 等)
        match = re.search(r"(\d+(\.\d+)?)", content)
        if match:
            score = float(match.group(1))
            # 确保分数在 0-1 之间
            score = max(0.0, min(1.0, score))
        else:
            print(f"LLM output format error: {content}")
            score = 0.0

    except Exception as e:
        print(f"API Call Failed: {e}")
        # 如果 API 失败，给一个默认低分，防止程序崩溃
        score = 0.0

    # 3. 存入缓存
    fitness_cache[mel_tuple] = score
    
    # 打印日志以便观察进度（因为 LLM 很慢）
    print(f"Evaluating Melody: Score {score}")
    
    return score

    

def run(alpha: float = 0.5,
        m =100,
        n = 10,
        crossover_probability = 0.1,
        mutation_probability = 0.1,
        transposition_probability = 0.05,
        retrograde_probability = 0.05,
        inversion_probability = 0.05,
        retrograde_inversion_probability = 0.03):
    """
    运行遗传算法

    alpha: 适应度阈值
    m:  迭代次数
    n: 种群大小
    crossover_probability: 交叉概率
    mutation_probability: 变异概率
    transposition_probability: 移调概率
    retrograde_probability: 逆行概率
    inversion_probability: 倒影概率
    retrograde_inversion_probability: 逆行倒影概率
    """

    # 1. 初始化种群
    random_generate_melodies = [generate_random_melody() for _ in range(n)]

    population = random_generate_melodies.copy()

    # 2. 迭代 m 次
    for _ in range(m):
        # 2.1 计算适应度
        with ThreadPoolExecutor(max_workers=n) as executor:
            fitnesses = list(executor.map(fitness, population))

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
                # 其他音乐变换操作：移调、逆行、倒影、逆行倒影
                selected_idx = roulette_wheel_selection(n, fitnesses)
                parent = population[selected_idx]
                
                # 随机选择一种音乐变换
                transform_choice = random.random()
                transform_sum = (transposition_probability + retrograde_probability + 
                               inversion_probability + retrograde_inversion_probability)
                
                if transform_sum > 0:
                    # 归一化概率
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
                    
                    next_generation.append(child)
                else:
                    # 如果所有变换概率都为0，直接复制父代
                    next_generation.append(parent.copy())
        population = next_generation[:n]

    # 3. 返回迭代后适应度 ≥ α 的旋律们 (按照适应度从大到小排列)
    result = sorted([mel for mel in population if fitness(mel) >= alpha], key=fitness, reverse=True)

    if result == None:
        print("没有找到适应度大于等于α的旋律")
    else:
        print("最高适应度：", fitness(result[0]))

    return result

if __name__ == "__main__":
    result = run(alpha=0.75, m=10000)
    print(result)
    for idx, mel in enumerate(result):
        output_path = f"./output/genetic_algorithm_result_{idx+1}.wav"
        synthesize_melody(
            codes=mel,
            output_path=output_path,
            sample_dir="./samples/",
            BPM=167,
            unit_time=180  
        )
