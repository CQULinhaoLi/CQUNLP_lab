import os
import glob
from config import Config as CFG

# 加载数据集
def load_imdb_data(data_dir="../imdbv1/aclImdb", is_train=True):
    """读取IMDb训练集, 返回(文本内容, 情感标签)列表"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    if is_train:
        data_dir = os.path.join(data_dir, 'train')
    else:
        data_dir = os.path.join(data_dir, 'test')
        
    data = []
    # 遍历正负样本文件夹
    # neg: 0, pos: 1
    for label, folder in enumerate(["neg", "pos"]):
        folder_path = os.path.join(data_dir, folder)
        # 匹配所有txt文件（排除隐藏文件）
        file_pattern = os.path.join(folder_path, "*.txt")
        for file_path in glob.glob(file_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    data.append( (text, label) )
            except UnicodeDecodeError:
                print(f"跳过损坏文件: {file_path}")
            except Exception as e:
                print(f"读取错误 {file_path}: {str(e)}")
    return data

# tokenize
import nltk
from nltk.tokenize import word_tokenize


def tokenize_corpus(corpus):
    nltk.download('punkt')
    tokenized_corpus = []
    for text, label in corpus:
        tokens = word_tokenize(text)
        tokenized_corpus.append((tokens, label))
    return tokenized_corpus


# build word dict
from collections import Counter

def build_vocab(tokenized_corpus, min_freq=5):
    """
    该函数用于根据分词后的语料库构建词表
    :param tokenized_corpus: 分词后的语料库，格式为 [(tokens, label), ...]
    :param min_freq: 词的最小出现频率，低于该频率的词将被过滤，默认为 5
    :return: 最终词表、词到索引的映射、索引到词的映射
    """
    all_tokens = []
    for tokens, _ in tokenized_corpus:
        all_tokens.extend(tokens)

    # 统计词频
    word_freq = Counter(all_tokens)

    # 构建词表
    vocab = sorted(word_freq, key=word_freq.get, reverse=True)

    # 过滤低频词
    filtered_vocab = [word for word in vocab if word_freq[word] >= min_freq]

    # 为词表添加特殊标记
    special_tokens = ['<PAD>', '<UNK>']
    final_vocab = special_tokens + filtered_vocab

    # 创建词到索引的映射
    word2idx = {word: idx for idx, word in enumerate(final_vocab)}
    idx2word = {idx: word for idx, word in enumerate(final_vocab)}

    return final_vocab, word2idx, idx2word


# mapping word to idx
def text_to_ids(tokenized_corpus, word2idx, unk_token="<UNK>"):
    """
    将分词语料转换为ID序列（含UNK处理）
    :param tokenized_corpus: 分词语料 [(tokens_list, label), ...]
    :param word2idx: 词表映射 {word: index}
    :param unk_token: 未登录词标记，默认"<UNK>"
    :return: (id_sequences, labels) 元组
    """
    unk_id = word2idx.get(unk_token, 0)  # 默认使用0作为UNK，需确保词表包含
    id_sequences = []
    labels = []
    
    for tokens, label in tokenized_corpus:
        # 转换单个样本：token → id，UNK处理
        seq_ids = [word2idx.get(token, unk_id) for token in tokens]
        
        # 简单校验（非空序列）
        if not seq_ids:
            print(f"警告：空序列，标签{label}, 原始tokens: {tokens}")
            continue
        
        id_sequences.append(seq_ids)
        labels.append(label)
    
    return id_sequences, labels


# 组装数据集，得到DataLoader
# ------------ 1. 定义PyTorch Dataset ------------
import torch
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
    def __init__(self, id_sequences, labels, pad_id=0):
        """
        :param id_sequences: train_ids/test_ids (列表的列表)
        :param labels: rain_labels/test_labels (列表)
        :param pad_id: <PAD>的ID (与词表一致, 默认0)
        """
        assert len(id_sequences) == len(labels), "数据与标签数量不匹配"
        self.seqs = id_sequences
        self.labels = labels
        self.pad_id = pad_id  # word2idx[<PAD>] = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]  # 返回原始未填充的序列和标签

# ------------ 2. 动态填充的Collate函数（关键！） ------------
def _imdb_collate(batch, pad_id=0):
    """
    动态填充批次, 返回适合RNN的(填充序列, 长度, 标签)
    :param batch: 原始批次数据 [(seq1, label1), (seq2, label2), ...]
    :return: (padded_seqs, seq_lengths, labels)
    """
    seqs, labels = zip(*batch)  # 解包批次
    
    # 计算每个序列的长度（用于RNN）
    seq_lengths = torch.LongTensor([len(seq) for seq in seqs])
    
    # 动态填充到批次最大长度
    max_len = seq_lengths.max().item()
    padded_seqs = []
    for seq in seqs:
        # 填充方式：后补pad_id（与你的text_to_ids逻辑一致）
        padded = seq + [pad_id] * (max_len - len(seq))
        padded_seqs.append(torch.LongTensor(padded))  # 转换为LongTensor
    
    # 堆叠为批次张量
    padded_seqs = torch.stack(padded_seqs)  # [B, L]
    labels = torch.LongTensor(labels)  # [B]
    
    return padded_seqs, seq_lengths, labels  # 直接用于RNN的pack操作

def get_data_loader(train_ids, train_labels, test_ids, test_labels, batch_size=CFG.batch_size, pad_id=0):
    """
    创建训练集和测试集的DataLoader
    :param train_ids: 训练集ID序列 (列表的列表)
    :param train_labels: 训练集标签 (列表)
    :param test_ids: 测试集ID序列 (列表的列表)
    :param test_labels: 测试集标签 (列表)
    :param batch_size: 每个批次的样本数量，默认32
    :param pad_id: 填充标记的ID，默认0
    :return: 训练集和测试集的DataLoader
    """
    # 创建训练集Dataset
    train_dataset = IMDBDataset(train_ids, train_labels, pad_id)
    # 创建训练集DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 每批次样本数量
        shuffle=True,  # 打乱数据顺序
        collate_fn=_imdb_collate,  # 使用动态填充的collate函数
        pin_memory=True  # 加速GPU数据加载
    )

    # 创建测试集Dataset
    test_dataset = IMDBDataset(test_ids, test_labels, pad_id)
    # 创建测试集DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # 每批次样本数量
        shuffle=False,  # 不打乱数据顺序
        collate_fn=imdb_collate  # 使用动态填充的collate函数
    )

    return train_loader, test_loader  # 返回训练集和测试集的DataLoader