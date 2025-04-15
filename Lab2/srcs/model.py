import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 num_layers=1, bidirectional=True, dropout=0.5, pad_idx=0):
        super(AttentionRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.attn_fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def attention(self, rnn_output, lengths):
        """
        注意力机制
        :param rnn_output: RNN的输出 [B, L, H]
        :param lengths: 每个样本的有效长度 [B]
        :return: 上下文向量和注意力权重
        """
        B, L, H = rnn_output.size()
        attn_scores = self.attn_fc(rnn_output).squeeze(-1)  # [B, L]

        # 构造 mask：[B, L]
        device = rnn_output.device
        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, L]

        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # 填充部分设置为极小值
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, L]

        # 上下文向量加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_output).squeeze(1)  # [B, H]
        return context, attn_weights

    def forward(self, text, lengths):
        """
        前向传播
        :param text: 输入文本 [B, L]
        :param lengths: 每个文本的有效长度 [B]
        :return: 分类输出和注意力权重
        """
        embedded = self.dropout(self.embedding(text))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H]

        context, attn_weights = self.attention(rnn_output, lengths)
        output = self.classifier(self.dropout(context))  # [B, output_dim]

        return output, attn_weights


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 num_layers=1, bidirectional=False, dropout=0.5, pad_idx=0):
        """
        情感分类RNN模型
        :param vocab_size: 词表大小
        :param embed_dim: 词嵌入维度
        :param hidden_dim: RNN隐藏层维度
        :param output_dim: 输出类别数（如正负情感）
        :param num_layers: RNN层数
        :param bidirectional: 是否使用双向RNN
        :param dropout: dropout概率
        :param pad_idx: 填充标记的索引，用于初始化嵌入层权重
        """
        super(SimpleRNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # LSTM层
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        
        # 如果是双向RNN，最后输出维度需乘以2
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 全连接层
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, lengths):
        """
        前向传播
        :param text: 输入文本 [B, L]
        :param lengths: 每个文本的原始长度 [B]
        :return: 分类输出 [B, output_dim]
        """
        # 嵌入层
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # 打包变长序列
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM层
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 获取最后一层的隐藏状态
        if self.lstm.bidirectional:
            # 双向RNN，拼接正向和反向的最后隐藏状态
            hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        else:
            # 单向RNN，取最后一层的隐藏状态
            hidden = self.dropout(hidden[-1])
        
        # 全连接层输出
        output = self.fc(hidden)  # [B, output_dim]
        return output
