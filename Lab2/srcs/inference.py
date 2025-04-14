import torch
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

def infer(model, sentence, word2idx, idx2word, device, pad_idx=0, max_len=100):
    model.eval()
    tokens = word_tokenize(sentence)
    unk_id = word2idx.get("<UNK>", 1)
    input_ids = [word2idx.get(token, unk_id) for token in tokens]

    # Padding
    length = len(input_ids)
    if length < max_len:
        input_ids += [pad_idx] * (max_len - length)
    else:
        input_ids = input_ids[:max_len]
        length = max_len

    input_tensor = torch.tensor([input_ids]).to(device)
    length_tensor = torch.tensor([length]).to(device)

    with torch.no_grad():
        logits, attn_weights = model(input_tensor, length_tensor)

    prediction = torch.argmax(logits, dim=1).item()
    return prediction, tokens, attn_weights.squeeze().cpu().numpy()
