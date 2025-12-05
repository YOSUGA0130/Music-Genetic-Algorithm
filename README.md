## 1. 环境与依赖

- 必需第三方库：
  - `pydub`（导出音频对比听感需要）

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 项目结构
```txt
project_root/
  audio_synth.py        # 根据编码数组拼合样本音频，生成完整旋律 wav
  note_encoding.py      # 处理旋律表示方式：JSON ↔ 内部数组编码
  random_melody.py      # 随机旋律生成（写入 Melody/Code & Melody/Audio）

  samples/
    1.wav               # 钢琴琴键音频，包含全部88键A0->C8
    2.wav
    ...
    88.wav

  Melody/
    Code/
      random_01.json   # 旋律的音名编码（可以查看和手动编辑）
      random_02.json
      ...
    Audio/
      random_01.wav    # 与上面 json 对应的合成音频
      random_02.wav
      ...
```

## 3. 当前采用的旋律表示方式

目前有两种表示方式：

##### 人类能读的 音名 表示（存放在 Melody/Code/）

- “0”：休止符
- “-”：延长符
- C5，#F4，bG3等：音名表示

适合手动调整、写最终报告或可视化说明www


##### 程序内部的数组编码（运行算法时使用）

使用一维整数数组表示旋律，用于遗传算法、随机生成等操作

- -1：延长符
- 0：休止符
- 1-88：对应A0->C8

##### 两种表示可以通过note_encoding.py中的note_to_int和int_to_note转换

整体流程示意：

JSON（给人看/改） → note_encoding.py → 内部整数数组 → audio_synth.py 合成 → .wav 音频

##### 旋律表示的要求
- 自选歌曲旋律：不限音域，不限节拍(?)，不限最小音符单元
- 随机生成旋律：$\frac{4}{4}$拍，每个json元素 & 每个数组元素 时值都是一个八分音符，一小节8个元素，共四小节

## 4.已经完成的部分内容
- Melody中包括15段随机旋律的wav和json音名编码，可以通过
```python
from note_encoding import note_to_int

    melody = note_to_int("/Melody/Code/random_0x.json") 
```
直接获得对应melody数组

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

crossover（交换）：随机位置和长度交换两对父旋律的片段，返回两个孩子

mutation（变异）：随机位置改变音符，返回新个体

## 5.一些问题
- 自选旋律时长很难统一😭，目前有三段自选旋律，大家可以先听个响www。似乎随机旋律和自选旋律只需要做一种(?) 先处理随机旋律，后续有时间再找些别的音乐也行()