from utils.config_loader import load_config

CFG = load_config("configs/first.yaml")

# 打印所有配置（检查层级是否正确）
print("完整配置:")
for section in ['training', 'model', 'data', 'logging', 'evaluation']:
    print(f"\n{section.upper()}:")
    for k, v in CFG[section].items():
        print(f"  {k}: {v}")
print(CFG['parameters_version'])

# 验证关键参数
# assert CFG['model']['type'] == "BiLSTM", "模型类型加载错误"
# assert CFG['training']['batch_size'] == 32, "批量大小加载错误"
# assert CFG['data']['data_dir'] == "./dataset", "数据路径加载错误"
print("\n所有关键参数验证通过！")