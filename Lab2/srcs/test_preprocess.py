from data_utils import load_imdb_data, tokenize_corpus, build_vocab, text_to_ids, get_data_loader
import os


def preprocess_data():
   
    data_dir = os.path.join("..", "imdb_aclImdb_v1", "aclImdb")
    # 1. 加载数据集
    train_corpus = load_imdb_data(data_dir, is_train=True)
    test_corpus = load_imdb_data(data_dir, is_train=False)

    # 2. 分词
    tokenized_train_corpus = tokenize_corpus(train_corpus)
    tokenized_test_corpus = tokenize_corpus(test_corpus)
    # for i in range(5):
    #     print(f"[{i}] tokens:", tokenized_train_corpus[i][0])
    #     print(f"[{i}] label:", tokenized_train_corpus[i][1])
    # 3. 构建词表
    vocab, word2idx, idx2word = build_vocab(tokenized_train_corpus, min_freq=3)

    # 4. 将文本转换为索引
    train_ids, train_labels = text_to_ids(tokenized_train_corpus, word2idx)
    test_ids, test_labels = text_to_ids(tokenized_test_corpus, word2idx)
    for i in range(5):
        print(f"[{i}] token ids:", train_ids[i])
        print(f"[{i}] length:", len(train_ids[i]))
    # 5. 创建数据集和数据加载器
    train_loader, test_loader = get_data_loader(train_ids, train_labels, test_ids, test_labels)

    print(f"词表大小（含特殊标记）：{len(vocab)}")
    print(f"<UNK> 示例 ID: {word2idx.get('<UNK>', -1)}")
    print(f"<PAD> 示例 ID: {word2idx.get('<PAD>', -1)}")

    # 看一下几个token是否能找到
    example_words = ["movie", "great", "bad", "awful"]
    for word in example_words:
        print(f"'{word}':", word2idx.get(word, "Not in vocab"))

    return train_loader, test_loader, vocab, word2idx, idx2word

preprocess_data()