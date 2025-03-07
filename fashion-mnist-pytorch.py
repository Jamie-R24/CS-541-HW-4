import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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

def findBestHyperparameters(X_train, y_train, X_val, y_val):
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    hidden_layer_sizes = [
        [784, 30, 10], [784, 40, 10], [784, 50, 10], 
        [784, 30, 30, 10], [784, 40, 40, 10], [784, 50, 50, 10], 
        [784, 30, 30, 30, 10], [784, 40, 40, 40, 10], [784, 50, 50, 50, 10]
    ]
    batch_sizes = [32, 64, 128, 256]
    
    best_val_acc = 0
    best_params = None

    with open('results2.txt', 'w') as f:
        f.write("Hyperparameter Search Results:\n")
        f.write("-----------------------------\n")
    
        for lr, hl_size, bs in product(learning_rates, hidden_layer_sizes, batch_sizes):
            f.write(f"\nTraining with learning rate: {lr}, hidden layers: {hl_size}, batch size: {bs}\n")
            model = FeedForwardNN(hl_size)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            num_epochs = 5  # Small number for quick hyperparameter tuning
            
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                permutation = torch.randperm(X_train.size()[0])
                
                for i in range(0, X_train.size()[0], bs):
                    indices = permutation[i:i + bs]
                    batch_x, batch_y = X_train[indices], y_train[indices]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    _, predicted = torch.max(val_outputs, 1)
                    val_acc = (predicted == y_val).sum().item() / y_val.size(0)
            
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = (lr, hl_size, bs)
                        best_model_state = model.state_dict()
            
            f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    
        f.write(f"\nBest hyperparameters: Learning rate: {best_params[0]}, Hidden layers: {best_params[1]}, Batch size: {best_params[2]}\n")
    
    return best_params, best_model_state

def train_and_evaluate():
    train_images = np.load('fashion_mnist_train_images.npy')
    train_labels = np.load('fashion_mnist_train_labels.npy')
    test_images = np.load('fashion_mnist_test_images.npy')
    test_labels = np.load('fashion_mnist_test_labels.npy')

    train_images = (train_images / 255.0) - 0.5
    test_images = (test_images / 255.0) - 0.5

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
    X_train, X_val, test_images = map(lambda x: torch.tensor(x, dtype=torch.float32), (X_train, X_val, test_images))
    y_train, y_val, test_labels = map(lambda y: torch.tensor(y, dtype=torch.long), (y_train, y_val, test_labels))

    best_params, best_model_state = findBestHyperparameters(X_train, y_train, X_val, y_val)
    
    final_model = FeedForwardNN(best_params[1])
    final_model.load_state_dict(best_model_state)
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params[0], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 30
    train_losses = []

    with open('results2.txt', 'a') as f:
        f.write("\nTraining with Best Hyperparameters:\n")
        f.write("-----------------------------\n")

        final_model.train()
        for epoch in range(num_epochs):
            permutation = torch.randperm(X_train.size()[0])
            
            for i in range(0, X_train.size()[0], best_params[2]):
                indices = permutation[i:i + best_params[2]]
                batch_x, batch_y = X_train[indices], y_train[indices]
                
                final_optimizer.zero_grad()
                outputs = final_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                final_optimizer.step()
            
            train_losses.append(loss.item())

            if epoch >= num_epochs - 20:
                f.write(f"Epoch {epoch}: Training loss = {loss.item():.4f}\n")
                print(f"Epoch {epoch}: Training loss = {loss.item():.4f}")

        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(test_images)
            _, predicted = torch.max(test_outputs, 1)
            test_acc = (predicted == test_labels).sum().item() / test_labels.size(0)
        
        f.write(f"\nTest Accuracy: {test_acc:.4f}\n")
        print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc, train_losses

if __name__ == "__main__":
    test_acc, train_losses = train_and_evaluate()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses[-20:])
    plt.title('Training Loss Over Last 20 Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss2.png')  # Save the plot as an image
    #plt.show()
