import matplotlib.pyplot as plt

def visualize_attention(tokens, attn_weights):
    fig, ax = plt.subplots(figsize=(len(tokens) * 0.6, 2))
    ax.bar(range(len(tokens)), attn_weights[:len(tokens)])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_ylabel("Attention Weight")
    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.show()
