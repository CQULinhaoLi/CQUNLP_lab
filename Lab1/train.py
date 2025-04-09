import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from data_loader import load_dataset
from tokenizer import convert_corpus_to_id
from mini_batch import split_data_set
from model import BiLSTMAttentionModel

def train_model(model, train_data, test_data, word_dict,
                label_dict, batch_size=32, max_seq_len=100,
                num_epochs=10, learning_rate=0.001):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the selected device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Define the optimizer
    criterion = nn.CrossEntropyLoss()  # Define the loss function

    for epoch in range(num_epochs):  # Loop over the number of epochs
        model.train()  # Set the model to training mode
        total_loss = 0  # Initialize total loss for the epoch

        # Iterate over batches of training data
        for batch in split_data_set(train_data, batch_size, max_seq_len, word_dict, label_dict):
            inputs, labels, text_lengths = batch  # Unpack the batch into inputs, labels, and text lengths
            inputs, labels = inputs.to(device), labels.to(device) # Move data to the selected device
            labels = labels.squeeze()
            print(labels)
            optimizer.zero_grad()  # Clear the gradients from the previous step
            outputs = model(inputs, text_lengths)  # Forward pass through the model with text_lengths
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate the loss
            break
        # Print the loss for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

        # Save the model for the current epoch in the 'model' folder
        torch.save(model.state_dict(), f'model/model_epoch_{epoch + 1}.pth')
        print(f"Model saved for epoch {epoch + 1} in the 'model' folder")

        # Evaluate the model on the test data
        model.eval()  # Set the model to evaluation mode
        correct = 0  # Initialize the count of correct predictions
        total = 0  # Initialize the total number of samples

        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch in split_data_set(test_data, batch_size, max_seq_len, word_dict, label_dict):
                inputs, labels, text_lengths = batch  # Unpack the batch into inputs, labels, and text lengths
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device

                outputs = model(inputs, text_lengths)  # Forward pass through the model with text_lengths
                _, predicted = torch.max(outputs, 1)  # Get the predicted class
                total += labels.size(0)  # Update the total number of samples
                correct += (predicted == labels).sum().item()  # Update the count of correct predictions

        # Compute and print the accuracy on the test data
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")



if __name__ == '__main__':
    # Load the dataset
    train_set, test_set, word_dict, label_dict = load_dataset('dataset/')

    # Convert the dataset into word and label IDs
    train_data = convert_corpus_to_id(train_set, word_dict, label_dict)
    test_data = convert_corpus_to_id(test_set, word_dict, label_dict)

    # Define model parameters
    vocab_size = len(word_dict)  # Vocabulary size
    embedding_dim = 128  # Embedding dimension
    hidden_dim = 64  # Hidden dimension for LSTM
    output_dim = len(label_dict)  # Number of output classes (labels)
    pad_idx = word_dict["[pad]"]  # Padding index

    # Initialize the model
    model = BiLSTMAttentionModel(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)

    # Train the model
    train_model(model, train_data, test_data, word_dict,
                label_dict, batch_size=64, max_seq_len=64,
                num_epochs=1, learning_rate=0.001)
    # Save the model
    # torch.save(model.state_dict(), 'model.pth')