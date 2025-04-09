import torch
import jieba
from data_loader import load_dataset
from model import Classifier

def load_model(model, model_path, model_name):
    full_model_path = f"{model_path}/{model_name}"
    checkpoint = torch.load(full_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {full_model_path}")
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
train_set, test_set, word_dict, label_dict = load_dataset(root_path)
id2label = dict([(item[1], item[0]) for item in label_dict.items()])  # Map label IDs to label names

# Initialize and load the model
model = Classifier()  # Replace with your actual model class
model_path = "./model"  # Replace with the actual path to your model file
model_name = "BilLSTM+AddictiveAttention.pdparams"  # Replace with your actual model name
model = load_model(model, model_path, model_name)

# Perform inference
title = "我爱北京天安门"
infer(model, title)