import os
import numpy as np
import matplotlib.pyplot as plt

from read_cifar import read_cifar, split_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    """
    Effectue une étape d'apprentissage sur le réseau de neurones avec la loss MSE.

    Paramètres :
        w1, b1, w2, b2 : Poids et biais des couches.
        data (np.ndarray) : Données d'entrée.
        targets (np.ndarray) : labels cibles.
        learning_rate (float) : Taux d'apprentissage.

    Retourne :
        Tuple : Poids et biais mis à jour ainsi que la valeur de la loss.
    """
    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = sigmoid(z1)  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = sigmoid(z2)  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer
    
    # Calcul de la loss (MSE)
    loss = np.mean(np.square(predictions - targets))
    
    # Backpropagation
    # Output layer gradients
    dL_da2 = 2 * (a2 - targets) / targets.shape[0]
    da2_dz2 = sigmoid_derivative(z2)
    
    dL_dz2 = dL_da2 * da2_dz2 # Using the chaining rule
    
    # Gradients pour w2 and b2
    dL_dw2 = np.matmul(a1.T, dL_dz2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    
    # Hidden layer gradients
    dL_da1 = np.matmul(dL_dz2, w2.T)
    da1_dz1 = sigmoid_derivative(z1)

    dL_dz1 = dL_da1 * da1_dz1
    
    # Gradients pour w1 and b1
    dL_dw1 = np.matmul(a0.T, dL_dz1)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    
    # Descente de gradient
    w1 -= learning_rate * dL_dw1
    b1 -= learning_rate * dL_db1
    w2 -= learning_rate * dL_dw2
    b2 -= learning_rate * dL_db2
    
    return w1, b1, w2, b2, loss

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(labels, num_classes):
    one_hot_matrix = np.zeros((labels.size, num_classes))
    one_hot_matrix[np.arange(labels.size), labels] = 1
    return one_hot_matrix

def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    """
    Effectue une étape d'apprentissage sur le réseau de neurones avec la cross entropy loss.

    Paramètres :
        w1, b1, w2, b2 : Poids et biais des couches.
        data (np.ndarray) : Données d'entrée.
        labels_train (np.ndarray) : labels de training.
        learning_rate (float) : Taux d'apprentissage.

    Retourne :
        Tuple : Poids et biais mis à jour ainsi que la valeur de la loss.
    """
    # Hyperparameters
    batch_size, d_in = data.shape
    d_out = w2.shape[1]  # Output dimension

    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (softmax activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = softmax(z2)  # output of the output layer (softmax activation function)

    # On convertit les labels en one-hot
    Y = one_hot(labels_train, d_out)
    
    # Cross-entropy loss
    loss = -np.mean(np.sum(Y * np.log(a2 + 1e-8), axis=1))

    # Backpropagation
    dZ2 = a2 - Y  
    dW2 = np.matmul(a1.T, dZ2) / batch_size
    dB2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

    dA1 = np.matmul(dZ2, w2.T)
    dZ1 = dA1 * a1 * (1 - a1)  
    dW1 = np.matmul(a0.T, dZ1) / batch_size
    dB1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

    # Descente de gradient
    w1 -= learning_rate * dW1
    b1 -= learning_rate * dB1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * dB2

    return w1, b1, w2, b2, loss

def calculate_accuracy(predictions, labels):
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == labels)

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    """
    Entraîne le réseau de neurones sur plusieurs epochs.

    Paramètres :
        w1, b1, w2, b2 : Poids et biais des couches.
        data_train, labels_train : Données et labels de training.
        learning_rate (float) : Taux d'apprentissage.
        num_epoch (int) : Nombre d'epochs de training.

    Retourne :
        Tuple : Poids, biais mis à jour, et les précisions de training à chaque epoch.
    """
    train_accuracies = []

    # On itère sur le nombre d'epochs
    for epoch in range(num_epoch):
        # On entraîne une fois
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        
        # Forward pass
        a0 = data_train 
        z1 = np.matmul(a0, w1) + b1
        a1 = 1 / (1 + np.exp(-z1))  
        z2 = np.matmul(a1, w2) + b2  
        predictions = softmax(z2)  

        # Mesure de la précision
        accuracy = calculate_accuracy(predictions, labels_train)
        train_accuracies.append(accuracy)
        
        # Print pour suivre l'avancement
        print(f"Epoch {epoch+1}/{num_epoch} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    return w1, b1, w2, b2, train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    """
    Teste le réseau de neurones sur un ensemble de test.

    Paramètres :
        w1, b1, w2, b2 : Poids et biais des couches.
        data_test (np.ndarray) : Données de test.
        labels_test (np.ndarray) : labels de test.

    Retourne :
        float : L'accuracy du réseau sur les données de test.
    """
    # Forward pass on test data
    a0 = data_test  # Input layer
    z1 = np.matmul(a0, w1) + b1  # Hidden layer pre-activation
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation
    z2 = np.matmul(a1, w2) + b2  # Output layer pre-activation
    predictions = softmax(z2)  # Softmax activation for classification

    # Calculate test accuracy
    test_accuracy = calculate_accuracy(predictions, labels_test)
    
    return test_accuracy

def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    # Define dimensions
    d_in = data_train.shape[1]  # Input dimension
    d_out = len(np.unique(labels_train))  # Output dimension (number of classes)

    # Initialize weights and biases
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # First layer weights
    b1 = np.zeros((1, d_h))  # First layer biases
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # Second layer weights
    b2 = np.zeros((1, d_out))  # Second layer biases

    train_accuracies = []

    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)

    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    
    return train_accuracies, test_accuracy

if __name__ == "__main__":

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Load CIFAR data, split into training and testing sets with split_factor=0.9
    data, labels = read_cifar("data/cifar-10-batches-py")
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split=0.9)

    # Define hyperparameters
    d_h = 64  # Number of neurons in the hidden layer
    learning_rate = 0.1
    num_epoch = 100

    # Run the MLP training
    train_accuracies, test_accuracy = run_mlp_training(
        data_train, labels_train, data_test, labels_test,
        d_h=d_h, learning_rate=learning_rate, num_epoch=num_epoch
    )

    # Plot training accuracy over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epoch + 1), train_accuracies, label="Training Accuracy", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("MLP Training Accuracy Across Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("results/mlp.png")
    plt.show()

    print("Final Test Accuracy:", test_accuracy)