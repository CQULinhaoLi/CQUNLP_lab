import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer, criterion, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = cfg.device
        self.cfg = cfg

        self.save_path = cfg.save_path
        self.use_early_stopping = getattr(cfg, 'use_early_stopping', True)
        self.early_stop_patience = getattr(cfg, 'early_stop_patience', 3)

        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

    def _train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss, correct, total = 0, 0, 0
        for x, y, lengths in tqdm(train_loader, desc="Training", leave=False):
            x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x, lengths)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return epoch_loss / len(train_loader), correct / total

    def _evaluate(self, val_loader):
        self.model.eval()
        epoch_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y, lengths in tqdm(val_loader, desc="Validating", leave=False):
                x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                outputs = self.model(x, lengths)
                loss = self.criterion(outputs, y)

                epoch_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return epoch_loss / len(val_loader), correct / total

    def fit(self, train_loader, val_loader):
        best_valid_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.cfg.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.cfg.num_epochs}")

            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._evaluate(val_loader)

            self.train_losses.append(train_loss)
            self.valid_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.valid_accs.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Valid Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Save model if improved
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print("✅ New best model saved!")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{self.early_stop_patience}")
                if self.use_early_stopping and patience_counter >= self.early_stop_patience:
                    print("⛔ Early stopping triggered.")
                    break

        self.plot_curves()

    def plot_curves(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.valid_losses, label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, label="Train Acc")
        plt.plot(epochs, self.valid_accs, label="Valid Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")

        plt.tight_layout()
        plt.show()
