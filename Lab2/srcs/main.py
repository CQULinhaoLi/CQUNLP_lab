from inference import infer
from visualize import visualize_attention
from model import AttentionRNN
import torch
from preprocess import preprocess_data
from config import Config as CFG
from train import train_model
# 读取并处理数据
print("加载数据集")
train_loader, test_loader, vocab, word2idx, idx2word = preprocess_data()
vocab_size = len(vocab)


# 训练
print("开始训练")
print("模型参数：", CFG.__dict__)
train_model(train_loader, test_loader, vocab_size, CFG)


# 加载模型
print("加载模型")
model = AttentionRNN(vocab_size=vocab_size, embed_dim=CFG.embed_dim, hidden_dim=CFG.hidden_dim,
                     output_dim=CFG.hidden_dim, num_layers=CFG.num_layers, bidirectional=CFG.bidirectional,
                     dropout=CFG.dropout)
model.load_state_dict(torch.load(CFG.best_model_path))
model.to(CFG.device)

# 推理 + 可视化
# 注意：这里的模型是经过训练的模型，推理时需要使用训练好的模型参数
print('开始推理')
sentence = "The movie was absolutely fantastic!"
label, tokens, attn_weights = infer(model, sentence, word2idx, idx2word, CFG.device, max_len=CFG.max_len)
print("Predicted label:", label)
visualize_attention(tokens, attn_weights)
