"""
Example script demonstrating MLflow experiment tracking with PyTorch models.
This script:
1. Sets up an MLflow experiment for a PyTorch neural network
2. Trains a neural network on the FashionMNIST dataset
3. Logs parameters, metrics, and models to MLflow
4. Tracks training progress and model architecture

Based on the [MLflow Deep Learning Guide](https://mlflow.org/docs/latest/deep-learning/pytorch/guide/index.html)
"""

import mlflow
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download and prepare the FashionMNIST dataset
# FashionMNIST is a dataset of Zalando's article images consisting of 60,000 training examples
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # Convert images to PyTorch tensors
)

# Create data loaders for efficient batch processing
# Batch size of 64 is a common choice for training neural networks
train_dataloader = DataLoader(training_data, batch_size=64)

# Check for GPU availability and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    """
    A simple feed-forward neural network for FashionMNIST classification.
    Architecture:
    - Input: 28x28 grayscale images (flattened to 784 features)
    - Hidden layers: Two fully connected layers with 512 units and ReLU activation
    - Output: 10 classes (FashionMNIST categories)
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # First hidden layer
            nn.ReLU(),                # ReLU activation
            nn.Linear(512, 512),      # Second hidden layer
            nn.ReLU(),                # ReLU activation
            nn.Linear(512, 10),       # Output layer (10 classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, metrics_fn, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        dataloader: DataLoader providing training batches
        model: Neural network to train
        loss_fn: Loss function (CrossEntropyLoss)
        metrics_fn: Metrics function (Accuracy)
        optimizer: Optimizer for updating model parameters
    """
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Move data to the appropriate device (CPU/GPU)
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log metrics every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch
            mlflow.log_metric("loss", f"{loss:3f}", step=(batch // 100))
            mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(batch // 100))
            print(
                f"loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(dataloader)}]"
            )


# Training configuration
epochs = 3
loss_fn = nn.CrossEntropyLoss()  # Standard loss function for classification
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = NeuralNetwork().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # Stochastic Gradient Descent

# Set up MLflow experiment
mlflow.set_experiment("torch_experiment_lab_1")

# Start MLflow run and track training
with mlflow.start_run():
    # Define and log training parameters
    params = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
    }
    mlflow.log_params(params)

    # Log model architecture summary
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}\n{'='*30}")
        train(train_dataloader, model, loss_fn, metric_fn, optimizer)

    # Save the trained model to MLflow for future use
    mlflow.pytorch.log_model(model, "model")
