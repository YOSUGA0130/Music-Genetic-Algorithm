## 1. 快速开始

### 克隆仓库：

```bash
git clone https://github.com/YOSUGA0130/Music-Genetic-Algorithm.git
cd Music-Genetic-Algorithm
```

### 安装依赖：

```bash
pip install -r requirements.txt
```

### 下载 LSTM 模型权重：

因为需要使用 LSTM 神经网络，请[ 点击此处 ](https://disk.pku.edu.cn/link/AACD3AE31B31214418A077A3F157923984)下载 LSTM 模型权重文件 `lstm_24.pth`，放置在 `model/` 目录下

### 运行主程序：

```bash
python3 main.py
```

三个可选适应度函数：

1. 规则算法

2. LSTM 模型

3. 优化后的规则算法

成品效果展示在[ 这里](https://disk.pku.edu.cn/link/AAEB250FC242B14783934B3EB93B256A86)

## 2. 项目结构

```txt
project_root/
├── config.py               # 全局配置参数
├── main.py                 # 机器作曲主入口
├── population.py           # 初始种群生成逻辑
├── genetics.py             # 遗传操作
├── fitness_rule.py         # 基于规则的适应度函数
├── fitness_rule_enhance.py # 优化后的基于规则的适应度函数
├── fitness_lstm.py         # 基于LSTM的适应度函数
├── fitness_llm.py          # 基于prompt工程的适应度函数
├── model/
│   ├── lstm.py             # LSTM模型定义
│   ├── lstm_24.pth         # LSTM模型权重文件
│   └── train.ipynb         # LSTM模型训练代码
├── util/
│   ├── audio_synth.py      # 音频合成工具
│   └── note_encoding.py    # 音符编码工具
├── samples/                # 钢琴采样文件
└── Melody/                 # 旋律数据
```
