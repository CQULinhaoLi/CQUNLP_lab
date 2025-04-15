from trainer import Trainer


import torch
# 训练模型函数修改：
def train_model(train_loader, test_loader, vocab_size, cfg, model_save_path, config_save_path):
    """
    训练模型
    :param train_loader: 训练集数据加载器
    :param test_loader: 测试集数据加载器
    :param model_save_path: 模型保存路径
    :param config_save_path: 配置保存路径
    """

    if cfg.use_attention:
        from model import AttentionRNN
        model = AttentionRNN(vocab_size=vocab_size,
                             embed_dim=cfg.embed_dim,
                             hidden_dim=cfg.hidden_dim,
                             output_dim=cfg.output_dim,
                             num_layers=cfg.num_layers,
                             bidirectional=cfg.bidirectional,
                             dropout=cfg.dropout)
    else:
        from model import SimpleRNN
        model = SimpleRNN(vocab_size=vocab_size,
                             embed_dim=cfg.embed_dim,
                             hidden_dim=cfg.hidden_dim,
                             output_dim=cfg.output_dim,
                             num_layers=cfg.num_layers,
                             bidirectional=cfg.bidirectional,
                             dropout=cfg.dropout)

    if cfg.criterion == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    trainer = Trainer(model, criterion, optimizer, cfg, model_save_path)
    trainer.fit(train_loader, test_loader)

    # 保存配置
    cfg.save_config(config_save_path)
    print(f"配置已保存至: {config_save_path}")