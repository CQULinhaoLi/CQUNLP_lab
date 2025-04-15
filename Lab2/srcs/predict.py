import os
import torch
import pickle
import json
from inference import infer
from visualize import visualize_attention
from model import AttentionRNN, SimpleRNN

# ========= 加载配置 =========
# 指定你要加载的模型时间戳
timestamp = "20250415-123835"  # <- 改为你要使用的版本
model_path = os.path.join("..", "saved_models", f"model_{timestamp}.pt")
config_path = os.path.join("..", "saved_models", f"config_{timestamp}.json")

print("加载配置文件")
with open(config_path, "r") as f:
    config_dict = json.load(f)

class CFG:
    pass

# 从配置文件动态添加属性
for k, v in config_dict.items():
    setattr(CFG, k, v)

CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= 加载数据 =========
print("加载数据集")
data_file = "preprocessed_data.pkl"
with open(data_file, "rb") as f:
    train_loader, test_loader, vocab, word2idx, idx2word = pickle.load(f)
print("数据已从文件加载")
vocab_size = len(vocab)

# ========= 构建模型并加载参数 =========
print("加载模型参数")
if CFG.use_attention:
    model = AttentionRNN(
        vocab_size=vocab_size,
        embed_dim=CFG.embed_dim,
        hidden_dim=CFG.hidden_dim,
        output_dim=CFG.output_dim,
        num_layers=CFG.num_layers,
        bidirectional=CFG.bidirectional,
        dropout=CFG.dropout,
    )
else:
    model = SimpleRNN(
        vocab_size=vocab_size,
        embed_dim=CFG.embed_dim,
        hidden_dim=CFG.hidden_dim,
        output_dim=CFG.output_dim,
        num_layers=CFG.num_layers,
        bidirectional=CFG.bidirectional,
        dropout=CFG.dropout,
    )

model.load_state_dict(torch.load(model_path, map_location=CFG.device))
model.to(CFG.device)
model.eval()

# ========= 推理和可视化 =========
print('开始推理')
sentence = "This movie is an absolute gem! With a gripping plot that hooks you from the start and outstanding performances breathing life into every character, its top - notch production quality immerses you fully. A must - watch!"
label, confidence, tokens, attn_weights = infer(model, sentence, word2idx, idx2word, CFG.device, max_len=CFG.max_len)

print(f"Predicted label: {label}, Confidence: {confidence * 100:.2f}%")
print("Tokens:", tokens)

if attn_weights is not None and CFG.visualize_attention:
    print("Attention weights:", attn_weights)
    visualize_attention(tokens, attn_weights)
