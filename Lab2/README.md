
# Lab2 电影评论情感分析

## 项目简介
基于PyTorch实现的电影评论情感分类任务，包含基础RNN模型与带注意力机制的改进版本，支持情感可视化分析。


## 目录结构
```bash
Lab2/
├── srcs/                   # 核心代码
│   ├── model.py            # RNN/AttentionRNN模型定义
│   ├── data_utils.py       # 数据集加载与预处理
│   ├── train.py            # 训练逻辑
│   ├── inference.py        # 推理与注意力可视化
│   ├── visualize.py        # 注意力权重可视化
│   ├── config.py           # 超参数配置
│   ├── main.py             # 训练入口
│   └── preprocessed_data.pkl # 预处理后的数据缓存
├── saved_models/           # 模型保存目录（自动生成）
```


## 环境依赖
```python
torch>=2.0          # 深度学习框架
nltk>=3.8          # 分词工具
matplotlib>=3.7    # 可视化
```


## 快速开始
### 1. 数据准备
- 下载数据集：[IMDB v1 数据集](https://www.kaggle.com/datasets/linhoalihx/imdbv1/data)

- 预处理完成后会生成`preprocessed_data.pkl`文件，供模型训练使用。
- 首次运行自动预处理并缓存：`python main.py`

### 2. 训练模型
```bash
python srcs/main.py  # 按config.py配置训练（默认开启注意力机制）
```
- 训练日志：自动保存最优模型到`saved_models/`
- 超参数配置：修改`srcs/config.py`中的`Config`类

### 3. 单条推理（含注意力可视化）
```python
from srcs.inference import infer
from srcs.visualize import visualize_attention

# 加载模型（示例）
model = AttentionRNN(...)
model.load_state_dict(...)

sentence = "This movie is fantastice！"
pred, confidence, tokens, attn_weights = infer(model, sentence, ...)
print(f"情感预测：{'正面' if pred==1 else '负面'}，置信度：{confidence:.2f}")
visualize_attention(tokens, attn_weights)  # 显示关键token的注意力权重
```


## 模型架构
### 1. 基础模型 (`SimpleRNN`)
- 嵌入层 → 双向LSTM（2层） → 全连接层
- 输入：词表索引序列（`[B, L]`）
- 输出：情感概率（`[B, 2]`）

### 2. 注意力增强模型 (`AttentionRNN`)
- 在LSTM输出后添加自注意力层
- 计算每个词的权重：`context = softmax(Wh + b) * h`
- 输出：加权上下文向量 + 注意力权重（用于可视化）


## 可视化示例
![注意力可视化](https://via.placeholder.com/400x200?text=Attention+Visualization)  
*（实际运行会生成Top 10高权重词汇的柱状图）*


## 配置说明
修改`srcs/config.py`调整训练参数：
```python
class Config:
    use_attention = True       # 是否启用注意力机制
    embed_dim = 256            # 词嵌入维度
    hidden_dim = 256           # LSTM隐藏层维度
    batch_size = 128           # 批量大小
    learning_rate = 1e-2       # 学习率
    max_len = 128              # 文本截断长度
    visualize_attention = True # 推理时是否可视化
```