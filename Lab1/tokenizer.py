import jieba

def convert_corpus_to_id(data_set, word_dict, label_dict):
    """
    Convert a dataset into word and label IDs using jieba for tokenization.

    Parameters:
        data_set (list of tuples): A list of (text, label) pairs.
        word_dict (dict): A dictionary mapping words to their IDs.
        label_dict (dict): A dictionary mapping labels to their IDs.

    Returns:
        list of tuples: A list of (word_ids, label_id) pairs.
    """
    result = []  # Initialize an empty list to store the converted dataset
    for text, label in data_set:  # Iterate through each (text, label) pair in the dataset
        # Tokenize the text using jieba and convert each word to its corresponding ID
        # If a word is not found in the word_dict, use the ID for "[oov]" (out-of-vocabulary)
        text = [word_dict.get(word, word_dict["[oov]"]) for word in jieba.cut(text)]
        # Convert the label to its corresponding ID using the label_dict
        result.append((text, label_dict[label]))
    # Return the list of (word_ids, label_id) pairs
    return result