"""
遗传算法
"""

from util.audio_synth import synthesize_melody
from util.note_encoding import int_to_note
import argparse
import sys
from genetics import (
    evolve, 
    CROSSOVER_PROBABILITY, 
    MUTATION_PROBABILITY, 
    TRANSPOSITION_PROBABILITY, 
    RETROGRADE_PROBABILITY, 
    INVERSION_PROBABILITY, 
    RETROGRADE_INVERSION_PROBABILITY
)
from population import generate_random_melody
from fitness_rule import fitness
from fitness_rule_enhance import run as run_enhance, fitness_enhanced
import numpy as np

def run(fitness_func,
        alpha: float = 0.5,
        m =100,
        n = 10,
        crossover_probability = CROSSOVER_PROBABILITY,
        mutation_probability = MUTATION_PROBABILITY,
        transposition_probability = TRANSPOSITION_PROBABILITY,
        retrograde_probability = RETROGRADE_PROBABILITY,
        inversion_probability = INVERSION_PROBABILITY,
        retrograde_inversion_probability = RETROGRADE_INVERSION_PROBABILITY):
    """
    运行遗传算法 (规则模式)
    参照课件 5 第 58 页代码框架
    """
    
    # 1. 初始化种群
    population = [generate_random_melody() for _ in range(n)]

    # 2. 迭代 m 次
    for i in range(m):
        # 2.1 计算适应度
        fitnesses = [fitness_func(mel) for mel in population]

        # 2.2 是否存在 ≥ α 的适应度的个体
        # 检查是否有满足条件的个体
        best_fitness = max(fitnesses)
        
        # 打印进度
        print(f"Gen {i}: Max Fitness = {best_fitness:.6f}")

        if best_fitness >= alpha:
            print(f"在第 {i} 代找到满足适应度 {alpha} 的个体，最高适应度: {best_fitness}")
            # 停机，返回满足条件的个体
            # 筛选并排序
            qualified = [(mel, fit) for mel, fit in zip(population, fitnesses) if fit >= alpha]
            qualified.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in qualified]

        # 2.3 进化生成下一代
        population = evolve(
            population, 
            fitnesses, 
            n,
            crossover_probability,
            mutation_probability,
            transposition_probability,
            retrograde_probability,
            inversion_probability,
            retrograde_inversion_probability
        )

    # 3. 达到最大迭代次数，返回结果
    print("达到最大迭代次数，未找到满足适应度 >= alpha 的旋律，返回当前最佳")
    
    fitnesses = [fitness_func(mel) for mel in population]
        
    return [mel for mel, fit in sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行机器作曲·遗传算法")
    parser.add_argument("--mode", type=str, choices=["rule", "lstm", "rule_enhance"], help="选择适应度评估模式: 'rule' (规则算法), 'lstm' (lstm神经网络) 或 'rule_enhance' (优化后的规则算法)")
    args = parser.parse_args()

    # 如果没有通过命令行指定模式，则交互式询问用户
    if args.mode is None:
        print("\n请选择适应度评估模式:")
        print("1. 规则算法")
        print("2. LSTM模型")
        print("3. 优化后的规则算法")
        choice = input("请输入选项 (1/2/3) [默认: 3]: ").strip()
        
        if choice == "1":
            mode = "rule"
        elif choice == "2":
            mode = "lstm"
        else:
            mode = "rule_enhance"
    else:
        mode = args.mode

    # 根据模式配置参数
    if mode == "lstm":
        # LSTM 模式: 直接调用 fitness_lstm.py 中的 run 函数 (因为一些机制实在不一样)
        from fitness_lstm import run as run_lstm
        result = run_lstm(alpha=0.3, m=100, n=2000)
    elif mode == "rule_enhance":
        # 优化后的规则模式：直接调用 fitness_rule_enhance.py 中的 run 函数，因为优化修改了框架
        result = run_enhance(alpha=0.85,n=20,look_ahead_steps=15, m=2000,fitness=fitness_enhanced,num_bars=8)
    else:
        result = run(fitness_func=fitness, alpha=0.9, m=10000, n=10)
    
    if not result:
        print("没有找到适应度大于等于α的旋律")
    else:
        print(f"生成了 {len(result)} 条旋律")
        print(result)
        
        for idx, mel in enumerate(result):
            if idx >= 10: break
            
            output_path = f"./output/{mode}/{mode}_result_{idx+1}.wav"
            synthesize_melody(
                codes=mel,
                output_path=output_path,
                sample_dir="./samples/",
                BPM=167,
                unit_time=180  
            )
            
            json_path = f"./output/{mode}/{mode}_result_{idx+1}.json"
            int_to_note(mel, json_path)
            print(f"已保存: {output_path} 和 {json_path}")
