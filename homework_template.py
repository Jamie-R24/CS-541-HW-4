import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.decomposition import PCA
import gzip
import os
import urllib.request


NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

def unpack(weightsAndBiases):
    Ws = []

    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

def forward_prop(x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    
    zs = []
    hs = [x]  
    
    for i in range(NUM_HIDDEN_LAYERS):
        z = np.dot(hs[i], Ws[i].T) + bs[i]
        zs.append(z)
        
        h = np.maximum(0, z)  
        hs.append(h)
    
    # Output layer
    z = np.dot(hs[-1], Ws[-1].T) + bs[-1]
    zs.append(z)
    
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    yhat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    hs.append(yhat)
    
    # Cross-entropy loss
    n_samples = x.shape[0]
    loss = -np.sum(y * np.log(np.clip(yhat, 1e-15, 1.0))) / n_samples
    
    return loss, zs, hs, yhat

def back_prop(x, y, weightsAndBiases):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    Ws, bs = unpack(weightsAndBiases)
    
    n_samples = x.shape[0]
    
    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases
    
    delta = yhat - y
    
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        dJdW = np.dot(delta.T, hs[i]) / n_samples
        dJdWs.insert(0, dJdW)
        
        dJdb = np.sum(delta, axis=0) / n_samples
        dJdbs.insert(0, dJdb)
        
        if i == 0:
            break
        
        delta = np.dot(delta, Ws[i])
        
        delta = delta * (zs[i-1] > 0)
    
    return np.hstack([dJdW.flatten() for dJdW in dJdWs] + [dJdb.flatten() for dJdb in dJdbs])

def train(trainX, trainY, weightsAndBiases, testX, testY):
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    n_samples = trainX.shape[0]
    trajectory = [weightsAndBiases.copy()]
    train_losses = []
    test_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Process mini-batches
        for start_idx in range(0, n_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = trainX[batch_indices]
            Y_batch = trainY[batch_indices]
            
            gradients = back_prop(X_batch, Y_batch, weightsAndBiases)
            
            # Update weights and biases
            weightsAndBiases -= LEARNING_RATE * gradients
        
        train_loss, _, _, train_preds = forward_prop(trainX, trainY, weightsAndBiases)
        train_losses.append(train_loss)
        
        trajectory.append(weightsAndBiases.copy())
        
        _, _, _, test_preds = forward_prop(testX, testY, weightsAndBiases)
        test_accuracy = np.mean(np.argmax(test_preds, axis=1) == np.argmax(testY, axis=1))
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return weightsAndBiases, trajectory

def initWeightsAndBiases():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

def plotSGDPath(trainX, trainY, trajectory):
    pca = PCA(n_components=2)
    
    trajectory_2d = pca.fit_transform(trajectory)
    
    grid_size = 20
    x_min, x_max = trajectory_2d[:, 0].min() - 1, trajectory_2d[:, 0].max() + 1
    y_min, y_max = trajectory_2d[:, 1].min() - 1, trajectory_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    
    Z = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            point_2d = np.array([[xx[i, j], yy[i, j]]])
            point_original = pca.inverse_transform(point_2d)
            
            loss, _, _, _ = forward_prop(trainX[:100], trainY[:100], point_original[0])
            Z[i, j] = loss
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    
    # Plot the loss surface
    surf = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.6)
    
    # Plot the SGD trajectory
    ax.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
              [forward_prop(trainX[:100], trainY[:100], w)[0] for w in trajectory],
              color='r', s=20, alpha=0.8)
    
    # colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Cross-Entropy Loss')
    ax.set_title('SGD Optimization Path Visualized in Weight Space')
    
    plt.show()

def load_fashion_mnist():
    """
    Load Fashion MNIST dataset
    """
    # URLs 
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data = {}
    
    for name, file in files.items():
        filepath = file
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
        
        # Read the data
        with gzip.open(filepath, 'rb') as f:
            if 'images' in name:
                # Skip header: magic number, number of images, rows, columns
                f.read(16)
                data[name] = np.frombuffer(f.read(), dtype=np.uint8)
                # For images: reshape and normalize
                if name == 'train_images':
                    data[name] = data[name].reshape(-1, 28*28)
                else:
                    data[name] = data[name].reshape(-1, 28*28)
            else:
                # Skip header: magic number, number of items
                f.read(8)
                data[name] = np.frombuffer(f.read(), dtype=np.uint8)
    
    return data

def preprocess_data(data):
    """
    Preprocess the Fashion MNIST data
    """
    train_images = data['train_images'].astype(np.float32) / 255.0 - 0.5
    test_images = data['test_images'].astype(np.float32) / 255.0 - 0.5
    
    train_labels = np.zeros((data['train_labels'].size, 10))
    train_labels[np.arange(data['train_labels'].size), data['train_labels']] = 1
    
    test_labels = np.zeros((data['test_labels'].size, 10))
    test_labels[np.arange(data['test_labels'].size), data['test_labels']] = 1
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    # Load and preprocess the Fashion MNIST dataset
    print("Loading and preprocessing Fashion MNIST dataset...")
    data = load_fashion_mnist()
    trainX, trainY, testX, testY = preprocess_data(data)
    
    print(f"Training data shape: {trainX.shape}, {trainY.shape}")
    print(f"Test data shape: {testX.shape}, {testY.shape}")
    
    print("Initializing weights and biases...")
    weightsAndBiases = initWeightsAndBiases()
    
    print("Performing gradient check...")
    grad_check_result = scipy.optimize.check_grad(
        lambda wab: forward_prop(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[:5]), wab)[0],
        lambda wab: back_prop(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[:5]), wab),
        weightsAndBiases
    )
    print(f"Gradient check result: {grad_check_result}")
    
    print("Training neural network...")
    weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)
    
    print("Plotting SGD path...")
    plotSGDPath(trainX, trainY, trajectory)
    
    _, _, _, test_preds = forward_prop(testX, testY, weightsAndBiases)
    final_accuracy = np.mean(np.argmax(test_preds, axis=1) == np.argmax(testY, axis=1))
    print(f"Final test accuracy: {final_accuracy:.4f}")