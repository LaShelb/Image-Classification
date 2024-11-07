import os
import numpy as np
import matplotlib.pyplot as plt

from read_cifar import read_cifar, split_dataset


def distance_matrix(matrix_a, matrix_b):

    A_squared = np.sum(matrix_a**2, axis=1).reshape(-1, 1)
    
    B_squared = np.sum(matrix_b**2, axis=1).reshape(1, -1)
    
    AB_product = np.dot(matrix_a, matrix_b.T)
    
    dists = np.sqrt(A_squared + B_squared - 2 * AB_product)

    return dists


def knn_predict(dists, labels_train, k):

    knn_indices = np.argsort(dists, axis=0)[:k, :]
    
    knn_labels = labels_train[knn_indices]

    predicted_labels = []
    
    for i in range(knn_labels.shape[1]):
        unique, counts = np.unique(knn_labels[:, i], return_counts=True)
        majority_label = unique[np.argmax(counts)]
        predicted_labels.append(majority_label)
    
    return predicted_labels

def evaluate_knn(data_train, labels_train, data_test, labels_test, k):

    dists = distance_matrix(data_train, data_test)
    
    predicted_labels = knn_predict(dists, labels_train, k)
    
    accuracy = np.mean(predicted_labels == labels_test)
    
    return accuracy


if __name__ == "__main__":

    directory_path = "data/cifar-10-batches-py"  
    data, labels = read_cifar(directory_path)

    split_factor = 0.9
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split_factor)

    k_values = range(1, 2)
    accuracies = [evaluate_knn(data_train, labels_train, data_test, labels_test, k) for k in k_values]

    # Tracer l'accuracy en fonction de k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Variation of Accuracy as a Function of k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Sauvegarder le graphique
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/knn.png")
    plt.show()