import torch
import json

class Config:
    # ========= 模型超参数 =========
    embed_dim = 256
    hidden_dim = 256
    output_dim = 2  # 二分类
    num_layers = 2
    bidirectional = True
    dropout = 0.2

    # ========= 训练超参数 =========
    batch_size = 128
    learning_rate = 1e-2
    num_epochs = 50
    early_stop_patience = 5

    # ========= 损失函数和优化器 =========
    criterion = "CrossEntropyLoss"  # 损失函数
    optimizer = "Adam"  # 优化器


    # ========= 数据参数 =========
    min_freq = 3  # 构建词表时的最小词频
    max_len = 128  # padding 长度上限

    # ========= 模型与日志路径 =========
    best_model_path = "../saved_models/best_RNN_attention_model.pt"

    # ========= 设备设置 =========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= 可选开关 =========
    first_run = True # 是否第一次运行
    use_attention = True # 是否使用注意力机制 对应AttentionRNN/SimpleRNN
    visualize_attention = True  # 推理时是否可视化注意力
    use_early_stopping = True  # 是否使用早停

    @staticmethod
    def save_config(save_path):
        config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)}
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=4)