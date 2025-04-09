import os
import jieba
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
import matplotlib.pyplot as plt


def load_dataset(path):
    # 读取数据集
    train_path = os.path.join(path, 'train.txt')
    test_path = os.path.join(path, 'test.txt')
    dict_path = os.path.join(path, 'dict.txt')
    label_path = os.path.join(path, 'label_dict.txt')
    
    # 加载词典
    with open(dict_path, 'r', encoding='utf-8') as f:
        # 读取词典文件并生成词典字典，字典中键为单词，值为对应单词在词典中的索引
        words = [word.strip() for word in f.readlines()]
        word_dict = dict(zip(words, range(len(words))))

    # 加载标签词典
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines()]
        # 将标签转换为数字
        # 例如：将“星座”转换为0，将“科技”转换为1
        lines = [(line[0], int(line[1])) for line in lines]
        label_dict = dict(lines)

    # 加载数据集
    def load_data(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label, text = line.strip().split("\t", maxsplit=1)
                data.append((text, label))
        return data
    
    train_set = load_data(train_path)
    test_set = load_data(test_path)

    return train_set, test_set, word_dict, label_dict
