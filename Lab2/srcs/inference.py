# import torch
# from nltk.tokenize import word_tokenize
# import matplotlib.pyplot as plt

# def infer(model, sentence, word2idx, idx2word, device, pad_idx=0, max_len=100):
#     model.eval()
#     tokens = word_tokenize(sentence)
#     unk_id = word2idx.get("<UNK>", 1)
#     input_ids = [word2idx.get(token, unk_id) for token in tokens]

#     # Padding
#     length = len(input_ids)
#     if length < max_len:
#         input_ids += [pad_idx] * (max_len - length)
#     else:
#         input_ids = input_ids[:max_len]
#         length = max_len

#     input_tensor = torch.tensor([input_ids]).to(device)
#     length_tensor = torch.tensor([length]).to(device)

#     with torch.no_grad():
#         logits, attn_weights = model(input_tensor, length_tensor)

#     prediction = torch.argmax(logits, dim=1).item()
#     return prediction, tokens, attn_weights.squeeze().cpu().numpy()
# import torch
# from nltk.tokenize import word_tokenize

# def infer(model, sentence, word2idx, idx2word, device, pad_idx=0, max_len=100):
#     model.eval()
#     tokens = word_tokenize(sentence)
#     unk_id = word2idx.get("<UNK>", 1)
#     input_ids = [word2idx.get(token, unk_id) for token in tokens]

#     # Padding
#     length = len(input_ids)
#     if length < max_len:
#         input_ids += [pad_idx] * (max_len - length)
#     else:
#         input_ids = input_ids[:max_len]
#         length = max_len

#     input_tensor = torch.tensor([input_ids]).to(device)
#     length_tensor = torch.tensor([length]).to(device)

#     with torch.no_grad():
#         output = model(input_tensor, length_tensor)

#         # 判断是否有 attention 返回值
#         if isinstance(output, tuple):
#             logits, attn_weights = output
#             attn_weights = attn_weights.squeeze().cpu().numpy()
#         else:
#             logits = output
#             attn_weights = None

#     prediction = torch.argmax(logits, dim=1).item()
#     return prediction, tokens, attn_weights

import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

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
        output = model(input_tensor, length_tensor)

        # 兼容 attention / 非 attention 模型
        if isinstance(output, tuple):
            logits, attn_weights = output
            attn_weights = attn_weights.squeeze().cpu().numpy()
        else:
            logits = output
            attn_weights = None

        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()

    return prediction, confidence, tokens, attn_weights

