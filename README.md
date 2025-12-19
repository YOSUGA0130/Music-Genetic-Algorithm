## 1. 快速开始

- 克隆仓库：

```bash
git clone https://github.com/YOSUGA0130/Music-Genetic-Algorithm.git
cd Music-Genetic-Algorithm
```

- 安装依赖：

```bash
pip install -r requirements.txt
```

- 如果要使用 LSTM 神经网络，请从[https://disk.pku.edu.cn/link/AACD3AE31B31214418A077A3F157923984](1)下载 LSTM 模型权重文件 `lstm_24.pth`，放置在 `model/` 目录下

- 运行主程序：

```bash
python3 main.py
```

## 2. 项目结构

```txt
project_root/
  config.py             # 全局配置参数
  main.py               # 机器作曲主入口
  population.py         # 初始种群生成逻辑
  genetics.py           # 遗传操作
  fitness_rule.py       # 基于规则的适应度函数
  fitness_lstm.py       # 基于LSTM的适应度函数
  fitness_llm.py        # 基于prompt工程的适应度函数

  model/
    lstm.py             # LSTM模型定义
    lstm_24.pth         # LSTM模型权重文件
    train.ipynb         # LSTM模型训练代码

  util/
    audio_synth.py      # 音频合成工具
    note_encoding.py    # 音符编码工具

  samples/              # 钢琴采样文件
  Melody/               # 旋律数据
```

## 3. 当前采用的旋律表示方式

目前有两种表示方式：

##### 人类能读的 音名 表示（存放在 Melody/Code/）

- “0”：休止符
- “-”：延长符
- C5，#F4，bG3 等：音名表示

适合手动调整、写最终报告或可视化说明 www

##### 程序内部的数组编码（运行算法时使用）

使用一维整数数组表示旋律，用于遗传算法、随机生成等操作

- -1：延长符
- 0：休止符
- 1-88：对应 A0->C8

##### 两种表示可以通过 note_encoding.py 中的 note_to_int 和 int_to_note 转换

流程示意：

JSON（可以查看/手动修改） → note_encoding.py → 内部整数数组 → audio_synth.py 合成 → .wav 音频

##### 旋律表示的要求

- 自选歌曲旋律：不限音域，不限节拍(?)，不限最小音符单元，长度目前取 8 小节
- 随机生成旋律：四四拍，每个 json 元素 & 每个数组元素 时值都是一个八分音符，每小节 8 个元素，共 4 小节

## 4.已经完成的部分内容

- Melody 中包括 15 段随机旋律的 wav 和 json 音名编码(random_n)，4 段自选旋律的 wav 和 json 音名编码(custom_n), 可以通过

```python
from note_encoding import note_to_int

    melody = note_to_int("Melody/Code/random_n.json") #或custom_n.json
```

直接获得对应 melody 数组。Melody/MelodyList.json 中可以查看所有可用旋律列表

- 可以通过数组合成音频

```python
from note_encoding import int_to_note
from audio_synth import synthesize_melody
        #假设有旋律数组mel
        int_to_note(mel, "Melody/Code/test.json")
        synthesize_melody(mel, "Melody/Audio/test.wav", sample_dir="samples", unit_time=180)
        #test.wav可以直接听
```

- 遗传算法部分

run：遗传算法主框架

roulette_wheel_selection 函数：轮盘赌选择

crossover（交换）：随机位置和长度交换两对父旋律的片段，返回两个孩子

mutation（变异）：随机位置改变音符，返回新个体

apply_and_check_transform 函数：应用音乐变换并检查超出范围的音符数量

is_transform_acceptable 函数：检查变换结果是否可接受

transposition（移调）：将旋律整体升高或降低若干半音

retrograde（逆行）：将旋律数组倒序

inversion（倒影）：以某个音高为轴心，将旋律上下翻转

retrograde_inversion（逆行倒影）：逆行倒影的复合变换

> 音乐变换操作后是这样处理的：
>
> - 如果超过一定比例的音符超出范围，重新进行变换，最多尝试一定次数，如果都不行，返回原旋律
> - 超出范围的音符会随机变成休止符(0)或延长符(-1)
> - 第一个位置不能是延长符，会强制变成休止符
> - 延长符只能跟在音符或延长符后面，不能跟在休止符后面

## 5.一些问题

- 目前有 4 段自选旋律，时长都是 8 小节(对于大部分歌曲而言，4 小节相对有点短了 www)；15 段随机旋律时长都是 4 小节。似乎随机旋律和自选旋律只需要做一种(?) 先处理随机旋律，后续有时间再找些别的音乐也行()

## 6.LSTM 模型

- 数据集：[sander-wood/melodyhub](https://huggingface.co/datasets/sander-wood/melodyhub)开源数据集中切分 4/4 拍 4 小节片段，共包含 322484 条训练集数据与 8363 条验证集数据。
- 适应度函数：使用负对数似然(Negative Log-Likelihood, NLL)作为损失函数进行训练，对于进化过程中每个染色体进行困惑度(Perplexity, PPL)计算，并 scale 到 0-1.

$$
Fitness = \frac{1}{PPL} = e^{-\text{NLL}} = \exp\left( \frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{< i}) \right)
$$
