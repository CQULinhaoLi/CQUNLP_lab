import torch
from model import Classifier

def test_classifier_forward():
    # Define test parameters
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    embedding_size = 16
    hidden_size = 32
    n_classes = 5
    n_layers = 2
    direction = 'bidirectional'
    dropout_rate = 0.1

    # Create random input data
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_lengths = torch.randint(1, seq_len + 1, (batch_size,))

    # Initialize the model
    model = Classifier(
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        n_classes=n_classes,
        n_layers=n_layers,
        direction=direction,
        dropout_rate=dropout_rate
    )

    # Forward pass
    logits = model(inputs, input_lengths)

    # Assertions
    assert logits.shape == (batch_size, n_classes), f"Expected logits shape {(batch_size, n_classes)}, but got {logits.shape}"
    print("Test passed: Classifier forward method works as expected.")

# Run the test
if __name__ == "__main__":
    test_classifier_forward()