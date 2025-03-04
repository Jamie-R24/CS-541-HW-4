import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float, alpha: float, batch_size: int):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.params = self.initialize_parameters()

    def initialize_parameters(self) -> dict:
        params = {}
        for i in range(len(self.layer_sizes) - 1):
            params[f'W{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            params[f'b{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))
        return params

    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1, 0)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[dict, np.ndarray]:
        cache = {'A0': X}
        A = X
        for i in range(len(self.layer_sizes) - 2):
            Z = np.dot(A, self.params[f'W{i+1}']) + self.params[f'b{i+1}']
            A = self.relu(Z)
            cache[f'Z{i+1}'] = Z
            cache[f'A{i+1}'] = A
        ZL = np.dot(A, self.params[f'W{len(self.layer_sizes) - 1}']) + self.params[f'b{len(self.layer_sizes) - 1}']
        AL = self.softmax(ZL)
        cache[f'Z{len(self.layer_sizes) - 1}'] = ZL
        cache[f'A{len(self.layer_sizes) - 1}'] = AL
        return cache, AL

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        N = y_true.shape[0]
        correct_logprobs = -np.log(y_pred[range(N), y_true])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * self.alpha * np.sum([np.sum(self.params[f'W{i+1}'] ** 2) for i in range(len(self.layer_sizes) - 1)])
        return data_loss + reg_loss

    def backward(self, cache: dict, y_true: np.ndarray) -> dict:
        grads = {}
        N = y_true.shape[0]
        y_one_hot = np.zeros((N, self.layer_sizes[-1]))
        y_one_hot[np.arange(N), y_true] = 1
        dZL = cache[f'A{len(self.layer_sizes) - 1}'] - y_one_hot
        grads[f'dW{len(self.layer_sizes) - 1}'] = np.dot(cache[f'A{len(self.layer_sizes) - 2}'].T, dZL) / N + self.alpha * self.params[f'W{len(self.layer_sizes) - 1}']
        grads[f'db{len(self.layer_sizes) - 1}'] = np.sum(dZL, axis=0, keepdims=True) / N
        dA_prev = np.dot(dZL, self.params[f'W{len(self.layer_sizes) - 1}'].T)
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            dZ = dA_prev * self.relu_derivative(cache[f'Z{i}'])
            grads[f'dW{i}'] = np.dot(cache[f'A{i-1}'].T, dZ) / N + self.alpha * self.params[f'W{i}']
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / N
            dA_prev = np.dot(dZ, self.params[f'W{i}'].T)
        return grads

    def update_parameters(self, grads: dict) -> None:
        for i in range(len(self.layer_sizes) - 1):
            self.params[f'W{i+1}'] -= self.learning_rate * grads[f'dW{i+1}']
            self.params[f'b{i+1}'] -= self.learning_rate * grads[f'db{i+1}']

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_epochs: int, verbose: bool = True) -> List[float]:
        num_train = X_train.shape[0]
        train_losses = []
        best_val_acc = 0
        best_params = None
        for epoch in range(num_epochs):
            indices = np.random.permutation(num_train)
            X_train = X_train[indices]
            y_train = y_train[indices]
            for i in range(0, num_train, self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                cache, AL = self.forward(X_batch)
                loss = self.compute_loss(y_batch, AL)
                train_losses.append(loss)
                grads = self.backward(cache, y_batch)
                self.update_parameters(grads)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {key: value.copy() for key, value in self.params.items()}
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")
        if best_params is not None:
            self.params = best_params
        return train_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, AL = self.forward(X)
        return np.argmax(AL, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        _, AL = self.forward(X)
        loss = self.compute_loss(y, AL)
        predictions = np.argmax(AL, axis=1)
        accuracy = np.mean(predictions == y)
        return loss, accuracy

    def check_grad(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float:
        _, AL = self.forward(X)
        grads = self.backward({'A0': X, f'A{len(self.layer_sizes) - 1}': AL}, y)
        num_params = len(self.layer_sizes) - 1
        for i in range(num_params):
            W = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']
            dW = np.zeros_like(W)
            db = np.zeros_like(b)
            for j in range(W.shape[0]):
                for k in range(W.shape[1]):
                    W[j, k] += epsilon
                    loss_plus = self.compute_loss(y, self.forward(X)[1])
                    W[j, k] -= 2 * epsilon
                    loss_minus = self.compute_loss(y, self.forward(X)[1])
                    W[j, k] += epsilon
                    dW[j, k] = (loss_plus - loss_minus) / (2 * epsilon)
            for j in range(b.shape[1]):
                b[0, j] += epsilon
                loss_plus = self.compute_loss(y, self.forward(X)[1])
                b[0, j] -= 2 * epsilon
                loss_minus = self.compute_loss(y, self.forward(X)[1])
                b[0, j] += epsilon
                db[0, j] = (loss_plus - loss_minus) / (2 * epsilon)
            grad_diff = np.linalg.norm(grads[f'dW{i+1}'] - dW) + np.linalg.norm(grads[f'db{i+1}'] - db)
        return grad_diff

def train_and_evaluate():
    train_images = np.load('fashion_mnist_train_images.npy')
    train_labels = np.load('fashion_mnist_train_labels.npy')
    test_images = np.load('fashion_mnist_test_images.npy')
    test_labels = np.load('fashion_mnist_test_labels.npy')

    with open('results.txt', 'w') as f:
        f.write("Fashion MNIST Classification Results\n")
        f.write("===================================\n\n")
        
        train_images = (train_images / 255.0) - 0.5
        test_images = (test_images / 255.0) - 0.5

        X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        batch_sizes = [32, 64, 128, 256]
        
        best_val_acc = 0
        best_params = None
        best_model = None
        
        f.write("Hyperparameter Search Results:\n")
        f.write("-----------------------------\n")

        for lr in learning_rates:
            for alpha in alphas:
                for batch_size in batch_sizes:
                    result = f"\nTrying lr={lr}, alpha={alpha}, batch_size={batch_size}"
                    print(result)
                    f.write(result + '\n')
                    
                    model = NeuralNetwork(
                        layer_sizes=[784, 128, 64, 10],
                        learning_rate=lr,
                        alpha=alpha,
                        batch_size=batch_size
                    )
  
                    model.train(X_train, y_train, X_val, y_val, num_epochs=5, verbose=False)

                    val_loss, val_acc = model.evaluate(X_val, y_val)
                    result = f"Validation accuracy: {val_acc:.4f}"
                    print(result)
                    f.write(result + '\n')
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            'learning_rate': lr,
                            'alpha': alpha,
                            'batch_size': batch_size
                        }
                        best_model = model
        
        result = "\nBest hyperparameters found:"
        print(result)
        f.write('\n' + result + '\n')
        for param, value in best_params.items():
            result = f"{param}: {value}"
            print(result)
            f.write(result + '\n')

        final_model = NeuralNetwork(
            layer_sizes=[784, 128, 64, 10],
            learning_rate=best_params['learning_rate'],
            alpha=best_params['alpha'],
            batch_size=best_params['batch_size']
        )

        train_losses = final_model.train(X_train, y_train, X_val, y_val, num_epochs=30)

        test_loss, test_acc = final_model.evaluate(test_images, test_labels)
        
        f.write('\nFinal Test Set Performance:\n')
        f.write('-------------------------\n')
        f.write(f"Cross-entropy loss: {test_loss:.4f}\n")
        f.write(f"Classification accuracy: {test_acc:.4f}\n")
        
        print(f"\nTest set performance:")
        print(f"Cross-entropy loss: {test_loss:.4f}")
        print(f"Classification accuracy: {test_acc:.4f}")
        
        f.write("\nResults have been saved to results.txt")
    
    return test_loss, test_acc, train_losses

if __name__ == "__main__":
    test_loss, test_acc, train_losses = train_and_evaluate()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')  # Save the plot as an image
    plt.show()
