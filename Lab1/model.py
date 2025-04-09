import torch
import torch.nn as nn
import torch.nn.functional as F

# class BiLSTMAttentionModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
#         super(BiLSTMAttentionModel, self).__init__()
#         # 定义嵌入层，vocab_size 是词汇表大小，embedding_dim 是嵌入维度，pad_idx 是填充标记的索引
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
#         # 定义双向 LSTM 层，hidden_dim 是隐藏层维度，bidirectional=True 表示双向
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
#         # 定义注意力机制的线性层，将 LSTM 输出映射到注意力权重
#         self.attention = nn.Linear(hidden_dim * 2, 1)
        
#         # 定义全连接层，将注意力后的上下文向量映射到输出类别
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
#         # 定义 Dropout 层，用于防止过拟合
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, text, text_lengths):
#         # 嵌入层：将输入的文本索引转换为嵌入向量，并应用 Dropout
#         embedded = self.dropout(self.embedding(text))
        
#         # 将嵌入向量打包为填充序列，text_lengths 是每个序列的实际长度
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        
#         # LSTM 层：处理打包的序列，返回打包的输出和隐藏状态
#         packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
#         # 解包序列：将打包的输出还原为填充序列
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
#         # 注意力机制：计算注意力权重
#         # 使用线性层将 LSTM 的输出映射到注意力分数，并通过 softmax 归一化
#         attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
        
#         # 计算上下文向量：将注意力权重与 LSTM 输出加权求和
#         context_vector = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        
#         # 全连接层：将上下文向量通过 Dropout 后映射到输出类别
#         logits = self.fc(self.dropout(context_vector))
        
#         return logits
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # 定义可学习的参数矩阵 w 和向量 v
        self.w = nn.Parameter(torch.empty(hidden_size, hidden_size))  # 权重矩阵
        self.v = nn.Parameter(torch.empty([1, hidden_size], dtype=torch.float32))  # 注意力向量

        # 初始化参数
        nn.init.xavier_uniform_(self.w)  # 使用 Xavier 初始化权重矩阵
        nn.init.xavier_uniform_(self.v)  # 使用 Xavier 初始化注意力向量

    def forward(self, inputs):
        """
        前向传播计算注意力机制
        :param inputs: 输入张量，形状为 [batch_size, seq_len, hidden_size]
        :return: 注意力加权后的上下文向量，形状为 [batch_size, hidden_size]
        """
        # 保存原始输入以便后续计算上下文向量
        last_layers_hiddens = inputs  # [batch_size, seq_len, hidden_size]

        # 转置输入以便与权重矩阵相乘
        inputs = torch.transpose(inputs, dim0=1, dim1=2)  # [batch_size, hidden_size, seq_len]

        # 计算注意力得分
        # 通过 tanh 激活函数对输入进行非线性变换
        inputs = torch.tanh(torch.matmul(self.w, inputs))  # [batch_size, hidden_size, seq_len]

        # 计算注意力权重
        # 将注意力向量与输入相乘并 squeeze 维度
        attn_weights = torch.matmul(self.v, inputs).squeeze(1)  # [batch_size, seq_len]

        # 对注意力权重进行 softmax 归一化
        attn_weights = F.softmax(attn_weights, dim=-1)  # [batch_size, seq_len]

        # 计算上下文向量
        # 将注意力权重与原始输入加权求和
        attn_vectors = torch.matmul(attn_weights.unsqueeze(1), last_layers_hiddens)  # [batch_size, 1, hidden_size]
        attn_vectors = torch.squeeze(attn_vectors, dim=1)  # [batch_size, hidden_size]

        return attn_vectors
    
class Classifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, n_classes=14, n_layers=1, direction='bidirectional', dropout_rate=0., init_scale=0.05):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.init_scale = init_scale

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,  # 词汇表大小
            embedding_dim=self.embedding_size,  # 嵌入维度
            weight=nn.Parameter(torch.empty(self.vocab_size, self.embedding_size).uniform_(-self.init_scale, self.init_scale))  # 嵌入权重
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,  # 输入维度
            hidden_size=self.hidden_size,  # 隐藏层维度
            num_layers=self.n_layers,  # LSTM 层数
            bidirectional=(direction == 'bidirectional'),  # 是否双向 LSTM
            batch_first=True,  # 输入和输出的第一个维度是 batch_size
            dropout=self.dropout_rate,  # Dropout 比例
        )

        self.dropout_emb = nn.Dropout(p=self.dropout_rate)  # 嵌入层的 Dropout

        self.attention = AttentionLayer(self.hidden_size * 2 if direction == 'bidirectional' else self.hidden_size)  # 注意力层
        self.cls_fc = nn.Linear(in_features=self.hidden_size*2 if direction == 'bidirectional' else self.hidden_size, out_features=self.n_classes) # 分类全连接层

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        embedded_input = self.embedding(inputs)
        if self.dropout_rate > 0.:
            embedded_input = self.dropout_emb(embedded_input)

        last_layers_hiddens, (last_step_hiddens, last_step_cells) = self.lstm(embedded_input)  # [batch_size, seq_len, hidden_size*2] or [batch_size, seq_len, hidden_size]

        attn_vectors = self.attention(last_layers_hiddens)

        logits = self.cls_fc(attn_vectors)

        return logits