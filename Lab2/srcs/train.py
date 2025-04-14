from trainer import Trainer
from config import Config as CFG

import torch

def train_model(train_loader, test_loader, vocab_size):
    """
    训练模型
    :param train_loader: 训练集数据加载器
    :param test_loader: 测试集数据加载器
    """

    if CFG.use_attention:
        from model import AttentionRNN
        model = AttentionRNN(vocab_size=vocab_size,
                             embed_dim=CFG.embed_dim,
                             hidden_dim=CFG.hidden_dim,
                             output_dim=CFG.output_dim,
                             num_layers=CFG.num_layers,
                             bidirectional=CFG.bidirectional,
                             dropout=CFG.dropout)
    else:
        from model import SimpleRNN
        model = SimpleRNN(vocab_size=vocab_size,
                             embed_dim=CFG.embed_dim,
                             hidden_dim=CFG.hidden_dim,
                             output_dim=CFG.output_dim,
                             num_layers=CFG.num_layers,
                             bidirectional=CFG.bidirectional,
                             dropout=CFG.dropout)

    if CFG.criterion == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    if CFG.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    elif CFG.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG.learning_rate)

    trainer = Trainer(model, criterion, optimizer, CFG)
    trainer.fit(train_loader, test_loader)
