import random
import numpy as np
def split_data_set(data_set, batch_size, max_seq_len, shuffle=True, drop_last=True, pad_id=1):
    """
    Splits the dataset into mini-batches.

    Parameters:
        data_set (list of lists): The dataset to be split, where each element is a sequence.
        batch_size (int): The number of sequences per batch.
        max_seq_len (int): The maximum length of sequences in a batch.
        shuffle (bool): Whether to shuffle the dataset before splitting. Default is True.
        drop_last (bool): Whether to drop the last batch if it is smaller than batch_size. Default is True.
        pad_id (int): The padding ID to use for sequences shorter than max_seq_len. Default is 1.

    Returns:
        list of lists: A list of mini-batches, where each mini-batch is a list of padded sequences.
    """
    if shuffle:
        # Shuffle the dataset to randomize the order of sequences
        random.shuffle(data_set)
    
    batch_text = []  # Temporary list to store sequences for the current batch
    batch_label = []  # Temporary list to store labels for the current batch

    for text, label in data_set:
        # Pad the sequence to max_seq_len with pad_id if it's shorter
        if len(text) < max_seq_len:
            text += [pad_id] * (max_seq_len - len(text))  # Add padding tokens
        else:
            text = text[:max_seq_len]  # Truncate the sequence if it's longer than max_seq_len
        
        # Ensure the sequence length matches max_seq_len
        assert len(text) == max_seq_len, f"Text length {len(text)} does not match max_seq_len {max_seq_len}."
        
        # Add the processed sequence and its label to the current batch
        batch_text.append(text)
        batch_label.append([label])

        # If the batch is full, yield it as a NumPy array and clear the batch lists
        if len(batch_text) == batch_size:
            yield np.array(batch_text).astype("int64"), np.array(batch_label).astype("int64")
            batch_text.clear()
            batch_label.clear()

    # If drop_last is False and there are leftover sequences, yield them as the last batch
    if (not drop_last) and len(batch_label) > 0:
        yield np.array(batch_text).astype("int64"), np.array(batch_label).astype("int64")
        batch_text.clear()
        batch_label.clear()