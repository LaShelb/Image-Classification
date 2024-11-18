import os
import numpy as np
import matplotlib.pyplot as plt

from read_cifar import read_cifar, split_dataset


def distance_matrix(matrix_a, matrix_b):
    """
    Calcule la matrice des distances euclidiennes entre deux matrices.

    Paramètres :
        matrix_a (np.ndarray) : Matrice A de taille (n_samples, n_features).
        matrix_b (np.ndarray) : Matrice B de taille (m_samples, n_features).

    Retourne :
        np.ndarray : Une matrice des distances de taille (n_samples, m_samples).
    """
    A_squared = np.sum(matrix_a**2, axis=1).reshape(-1, 1)
    
    B_squared = np.sum(matrix_b**2, axis=1).reshape(1, -1)
    
    AB_product = np.dot(matrix_a, matrix_b.T)
    
    dists = np.sqrt(A_squared + B_squared - 2 * AB_product)

    return dists


def knn_predict(dists, labels_train, k):
    """
    Prédit les labels pour un ensemble de données de test en utilisant le KNN.

    Paramètres :
        dists (np.ndarray) : Matrice des distances entre les données de training et de test.
        labels_train (np.ndarray) : labels de training.
        k (int) : Nombre de voisins à considérer.

    Retourne :
        List[int] : Les labels prédits pour chaque donnée de test.
    """
    # On trie les distances sur les colonnes (pour chaque donnée de test, on trie les données de training par distance)
    # Puis on récupère que les k premières lignes (qui sont les k premiers voisins)
    knn_indices = np.argsort(dists, axis=0)[:k, :]
    
    # On récupère les labels associés aux k premiers voisins pour chaque donnée de test
    knn_labels = labels_train[knn_indices]

    predicted_labels = []
    
    # On itère sur chaque donnée de test
    for i in range(knn_labels.shape[1]):
        # On trouve le label majoritaire sur la colonne
        unique, counts = np.unique(knn_labels[:, i], return_counts=True)
        majority_label = unique[np.argmax(counts)]
        predicted_labels.append(majority_label)
    
    return predicted_labels

def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    """
    Évalue le modèle KNN sur un ensemble de test.

    Paramètres :
        data_train (np.ndarray) : Données de training.
        labels_train (np.ndarray) : labels de training.
        data_test (np.ndarray) : Données de test.
        labels_test (np.ndarray) : labels de test.
        k (int) : Nombre de voisins à considérer.

    Retourne :
        float : L'accuracy du modèle sur les données de test.
    """
    # On evalue la distance entre les données de test et les données de training
    dists = distance_matrix(data_train, data_test)
    
    # On prédit les labels avec la méthode KNN
    predicted_labels = knn_predict(dists, labels_train, k)
    
    # On mesure le nombre de fois où la prediction est bonne
    accuracy = np.mean(predicted_labels == labels_test)
    
    return accuracy


if __name__ == "__main__":

    directory_path = "data/cifar-10-batches-py"  
    data, labels = read_cifar(directory_path)

    split_factor = 0.9
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split_factor)

    k_values = range(1, 20)
    accuracies = [evaluate_knn(data_train, labels_train, data_test, labels_test, k) for k in k_values]

    # On trace l'accuracy en fonction de k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Variation of Accuracy as a Function of k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # On sauvegarde le graphique
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/knn.png")
    plt.show()