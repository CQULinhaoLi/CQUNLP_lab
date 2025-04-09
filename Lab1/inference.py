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
    print("Predicted Label: ", predicted_label)

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
titles = [
    # ("星座", "男生眼里最有魅力的星座女"),
    # ("财经", "黄金未必比不动产更可靠"),
    # ("游戏", "网游《三国战魂》今日时技术三测"),
    # ("娱乐", "赵雅芝为肖像权讨万元，使用方称使用是因喜欢"),
    # ("体育", "米体：琼托利开转会会议，尤文有意奥斯梅恩、托纳利&科穆佐"),
    ("体育", "你还记得卡恩的这个骚操作吗？"),
    ("娱乐", "霍思燕黑羽毛裙亮相，雪白肌肤成焦点"),
]
for title in titles:
    print(f"Title: {title[1]}")
    print(f"True label: {title[0]}")
    infer(model, title[1])