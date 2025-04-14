from data_utils import load_imdb_data, tokenize_corpus, build_vocab, text_to_ids, get_data_loader
import os
from config import Config as CFG


def preprocess_data():
   
    data_dir = os.path.join("..", "imdb_aclImdb_v1", "aclImdb")
    # 1. 加载数据集
    train_corpus = load_imdb_data(data_dir, is_train=True)
    test_corpus = load_imdb_data(data_dir, is_train=False)

    # 2. 分词
    tokenized_train_corpus = tokenize_corpus(train_corpus)
    tokenized_test_corpus = tokenize_corpus(test_corpus)

    # 3. 构建词表
    vocab, word2idx, idx2word = build_vocab(tokenized_train_corpus, min_freq=5)

    # 4. 将文本转换为索引
    train_ids, train_labels = text_to_ids(tokenized_train_corpus, word2idx)
    test_ids, test_labels = text_to_ids(tokenized_test_corpus, word2idx)

    # 5. 创建数据集和数据加载器
    train_loader, test_loader = get_data_loader(train_ids, train_labels, test_ids, test_labels)


    return train_loader, test_loader, vocab, word2idx, idx2word
