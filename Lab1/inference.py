import torch
import jieba
from data_loader import load_dict
from model import Classifier

def load_model(model, model_path, model_name="model"):
    """
    Loads the trained model and optimizer state from the specified path with the given model name.
    """
    model_load_path = f"{model_path}/{model_name}.pdparams"
    
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    
    print(f"Model loaded from {model_load_path}")

    return model

def infer(model, text):
    model.eval()
    tokens = [word_dict.get(word, word_dict["[oov]"]) for word in jieba.cut(text)]
    tokens = torch.LongTensor(tokens).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(tokens)
        _, predicted = torch.max(output, 1)

    max_label_id = torch.argmax(output, dim=1).item()
    predicted_label = id2label[max_label_id]
    print("Label: ", predicted_label)

# Load dataset and prepare mappings
root_path = './dataset/'
word_dict, label_dict = load_dict(root_path)
id2label = dict([(item[1], item[0]) for item in label_dict.items()])  # Map label IDs to label names

vocab_size = len(word_dict.keys())
hidden_size = 128
embedding_size = 128
n_classes = len(label_dict.keys())
max_seq_len = 32
n_layers = 1
direction = 'bidirectional'
dropout_rate = 0.2


# Initialize and load the model
model = Classifier(
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        n_classes=n_classes,
        n_layers=n_layers,
        direction=direction,
        dropout_rate=dropout_rate
    )  
model_path = "./model"  
model_name = "BilLSTM+AddictiveAttention"
model = load_model(model, model_path, model_name)

# Perform inference
title = "云南发现恐龙新属种：金沙江元谋盗龙"
infer(model, title)