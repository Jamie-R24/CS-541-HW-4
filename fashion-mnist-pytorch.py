import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, layer_sizes):
        super(FeedForwardNN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_and_evaluate_pytorch():
    # Load data
    train_images = np.load('fashion_mnist_train_images.npy')
    train_labels = np.load('fashion_mnist_train_labels.npy')
    test_images = np.load('fashion_mnist_test_images.npy')
    test_labels = np.load('fashion_mnist_test_labels.npy')
    
    train_images = (train_images / 255.0) - 0.5
    test_images = (test_images / 255.0) - 0.5
    
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
    X_train, X_val, test_images = map(lambda x: torch.tensor(x, dtype=torch.float32), (X_train, X_val, test_images))
    y_train, y_val, test_labels = map(lambda y: torch.tensor(y, dtype=torch.long), (y_train, y_val, test_labels))
    
    # Define the model
    layer_sizes = [784, 128, 64, 10]
    model = FeedForwardNN(layer_sizes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 30
    batch_size = 64
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_val).sum().item() / y_val.size(0)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Validation Accuracy = {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images)
        _, predicted = torch.max(test_outputs, 1)
        test_acc = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc

if __name__ == "__main__":
    test_acc = train_and_evaluate_pytorch()
