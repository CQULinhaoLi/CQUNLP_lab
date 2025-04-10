import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data_loader import load_dataset
from tokenizer import convert_corpus_to_id
from mini_batch import split_data_set
from model import Classifier
from utils.config_loader import load_config


class Metric:
    """
    支持多类分类的指标计算类，提供单类指标和宏平均结果
    """
    def __init__(self, id2label):
        self.id2label = id2label  # 标签ID到名称的映射（如 {0: "cat", 1: "dog"}）
        self.reset()

    def reset(self):
        """按类别独立存储统计量"""
        self.class_stats = {}  # 格式：{label_id: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}
        self.total_samples = 0
        self.correct_samples = 0

    def update(self, real_labels, pred_labels):
        """
        real_labels: 真实标签数组（如 [0, 1, 0]）
        pred_labels: 预测标签数组（如 [0, 0, 1]）
        """
        self.total_samples += len(real_labels)
        self.correct_samples += np.sum(real_labels == pred_labels)

        # 获取所有可能的标签（包括真实和预测中出现的标签）
        all_labels = np.union1d(np.unique(real_labels), np.unique(pred_labels))

        for label in all_labels:
            # 初始化类别统计
            if label not in self.class_stats:
                self.class_stats[label] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            
            # 计算TP/FP/FN/TN
            tp = np.sum((real_labels == label) & (pred_labels == label))
            fp = np.sum((real_labels != label) & (pred_labels == label))
            fn = np.sum((real_labels == label) & (pred_labels != label))
            tn = np.sum((real_labels != label) & (pred_labels != label))

            # 累加统计量
            self.class_stats[label]["tp"] += tp
            self.class_stats[label]["fp"] += fp
            self.class_stats[label]["fn"] += fn
            self.class_stats[label]["tn"] += tn

    def get_class_result(self, label_id):
        """获取单个类别的指标"""
        if label_id not in self.class_stats:
            return None
        stats = self.class_stats[label_id]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "label": self.id2label[label_id],
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def get_macro_result(self):
        """获取宏平均指标（忽略无样本的类别）"""
        valid_classes = [c for c in self.class_stats if self.class_stats[c]["tp"] + self.class_stats[c]["fn"] > 0]
        if not valid_classes:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        precision = np.mean([self.get_class_result(c)["precision"] for c in valid_classes])
        recall = np.mean([self.get_class_result(c)["recall"] for c in valid_classes])
        f1 = np.mean([self.get_class_result(c)["f1"] for c in valid_classes])
        return {
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }

    def get_accuracy(self):
        """总体准确率"""
        return self.correct_samples / self.total_samples if self.total_samples > 0 else 0.0

    def format_print(self):
        """格式化输出（含单类指标和宏平均）"""
        print(f"{'Label':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}")
        print("-" * 45)
        for label_id in self.class_stats:
            res = self.get_class_result(label_id)
            print(f"{res['label']:<10} {res['precision']:.4f} || {res['recall']:.4f} || {res['f1']:.4f}")
        print("\nMacro Average:")
        macro_res = self.get_macro_result()
        print(f"Precision: {macro_res['macro_precision']:.4f}")
        print(f"Recall: {macro_res['macro_recall']:.4f}")
        print(f"F1: {macro_res['macro_f1']:.4f}")
        print(f"\nOverall Accuracy: {self.get_accuracy():.4f}")


def evaluate(model, test_set, word_dict, id2label, batch_size, max_seq_len, device):
    """
    Evaluates the model on the test set and prints the accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    metric = Metric(id2label)

    # Iterate over the test set in batches
    for batch_texts, batch_labels, batch_lengths in split_data_set(test_set, batch_size, max_seq_len, shuffle=False, pad_id=word_dict["[pad]"]):
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device).squeeze()

        with torch.no_grad():  # Disable gradient computation for evaluation
            logits = model(batch_texts, batch_lengths)
            _, pred_labels = torch.max(logits, dim=1)  # Get predicted labels

        # Update the metric with real and predicted labels
        metric.update(batch_labels.cpu().numpy(), pred_labels.cpu().numpy())

    # Get and print the evaluation results
    metric.format_print()

loss_records = []  # Global list to track loss values during training

def train(model, train_set, word_dict, id2label, optimizer, n_epochs, batch_size, max_seq_len, device):
    """
    Trains the model on the training set and evaluates it after each epoch.
    """
    model.train()  # Set the model to training mode
    global_step = 0  # Counter for global training steps
    metric = Metric(id2label)

    for epoch in range(n_epochs):
        model.train()   # Ensure the model is in training mode
        for step, (batch_texts, batch_labels, batch_lengths) in enumerate(split_data_set(train_set, batch_size, max_seq_len, shuffle=True, pad_id=word_dict["[pad]"])):
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device).squeeze()

            optimizer.zero_grad()  # Clear gradients from the previous step
            logits = model(batch_texts, batch_lengths)  # Forward pass
            _, pred_labels = torch.max(logits, dim=1)  # Get predicted labels

            # Compute the cross-entropy loss
            loss = F.cross_entropy(logits, batch_labels, reduction='mean')
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            # Log loss every 200 steps
            if step % 200 == 0:
                loss_records.append((global_step, loss.item()))
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            
            # Update the metric with real and predicted labels
            metric.update(batch_labels.cpu().numpy(), pred_labels.cpu().numpy())
            global_step += 1

        # Print training metrics after each epoch
        metric.format_print()
        save_model(classifier, optimizer, './model/first', model_name=f'{epoch+1}_{CFG['parameters_version']}') 


def show_loss_records():
    """
    Displays the recorded loss values during training.
    """
    import matplotlib.pyplot as plt

    x, y = zip(*loss_records)  # Unzip the loss records into x (steps) and y (loss values)
    plt.plot(x, y, label='Loss')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps')
    plt.legend()
    plt.show()

def save_model(model, optimizer, path, model_name="model"):
    """
    Saves the trained model and optimizer state to the specified path with the given model name.
    """
    model_save_path = f"{path}/{model_name}.pdparams"
    optimizer_save_path = f"{path}/{model_name}.optparams"
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Optimizer saved to {optimizer_save_path}")

if __name__ == '__main__':
    # Load dataset and preprocess
    CFG = load_config("configs/first.yaml")
    root_path = CFG['data']['data_dir']
    train_set, test_set, word_dict, label_dict = load_dataset(root_path)
    train_set = convert_corpus_to_id(train_set, word_dict, label_dict)
    test_set = convert_corpus_to_id(test_set, word_dict, label_dict)
    id2label = dict([(item[1], item[0]) for item in label_dict.items()])  # Map label IDs to label names

    # Hyperparameters
    n_epochs = CFG['training']['epochs']
    vocab_size = len(word_dict.keys())
    batch_size = CFG['training']['batch_size']
    hidden_size = CFG['model']['hidden_size']
    embedding_size = CFG['model']['embedding_size']
    n_classes = len(label_dict.keys())
    max_seq_len = CFG['model']['max_seq_len']  # Maximum sequence length for padding
    n_layers = CFG['model']['n_layers']
    dropout_rate = CFG['model']['dropout_rate']
    learning_rate = CFG['training']['learning_rate']
    direction = CFG['model']['direction']  # 'bidirectional' or 'unidirectional'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    print(f"Device: {device}")

    # Initialize the model
    classifier = Classifier(
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        n_classes=n_classes,
        n_layers=n_layers,
        direction=direction,
        dropout_rate=dropout_rate
    ).to(device)  # Move model to the appropriate device

    # Initialize the optimizer
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        betas=CFG['training']['betas'],
        weight_decay=CFG['training']['weight_decay'],
    )
    # Start training
    train(classifier, train_set, word_dict, id2label, optimizer, n_epochs, batch_size, max_seq_len, device)  # Train the model
    evaluate(classifier, test_set, word_dict, id2label, batch_size, max_seq_len, device)  # Evaluate the model
    show_loss_records()  # Show the loss records after training
    save_model(classifier, optimizer, './model/', model_name=CFG['parameters_version'])  # Save the trained model and optimizer state
