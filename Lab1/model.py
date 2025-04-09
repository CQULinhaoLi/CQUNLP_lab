import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(BiLSTMAttentionModel, self).__init__()
        # 定义嵌入层，vocab_size 是词汇表大小，embedding_dim 是嵌入维度，pad_idx 是填充标记的索引
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 定义双向 LSTM 层，hidden_dim 是隐藏层维度，bidirectional=True 表示双向
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # 定义注意力机制的线性层，将 LSTM 输出映射到注意力权重
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 定义全连接层，将注意力后的上下文向量映射到输出类别
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # 定义 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, text_lengths):
        # 嵌入层：将输入的文本索引转换为嵌入向量，并应用 Dropout
        embedded = self.dropout(self.embedding(text))
        
        # 将嵌入向量打包为填充序列，text_lengths 是每个序列的实际长度
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM 层：处理打包的序列，返回打包的输出和隐藏状态
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 解包序列：将打包的输出还原为填充序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 注意力机制：计算注意力权重
        # 使用线性层将 LSTM 的输出映射到注意力分数，并通过 softmax 归一化
        attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
        
        # 计算上下文向量：将注意力权重与 LSTM 输出加权求和
        context_vector = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        
        # 全连接层：将上下文向量通过 Dropout 后映射到输出类别
        logits = self.fc(self.dropout(context_vector))
        
        return logits
    