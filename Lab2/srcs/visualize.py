import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(tokens, attn_weights, top_k=10):
    # 转换为 numpy，确保兼容排序
    attn_weights = np.array(attn_weights)
    tokens = np.array(tokens)

    # 获取 attention 权重最大的前 top_k 个索引
    top_indices = np.argsort(attn_weights)[-top_k:][::-1]  # 从大到小排序

    # 选出对应的 token 和 attention 权重
    top_tokens = tokens[top_indices]
    top_weights = attn_weights[top_indices]

    # 绘图
    fig, ax = plt.subplots(figsize=(top_k * 0.6, 2))
    ax.bar(range(top_k), top_weights)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(top_tokens, rotation=45)
    ax.set_ylabel("Attention Weight")
    plt.title(f"Top {top_k} Tokens by Attention")
    plt.tight_layout()
    plt.show()
