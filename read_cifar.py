import os
import pickle
import numpy as np

def read_cifar_batch(path_batch: str):
    """
    Lit un lot de données CIFAR-10 à partir d'un fichier donné.

    Paramètres :
        path_batch (str) : Le chemin vers le fichier contenant le lot.

    Retourne :
        Tuple[np.ndarray, np.ndarray] : Une paire contenant les données du lot sous forme de tableau NumPy et les étiquettes associées.
    """
    with open(path_batch, 'rb') as f:

        batch = pickle.load(f, encoding='bytes')
        
    data = batch[b'data']
    
    labels = batch[b'labels']
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    return data, labels

def read_cifar(directory_path: str):
    """
    Lit tous les lots de données CIFAR-10 à partir d'un répertoire donné.

    Paramètres :
        directory_path (str) : Le chemin vers le répertoire contenant les lots.

    Retourne :
        Tuple[np.ndarray, np.ndarray] : Une paire contenant toutes les données concaténées et leurs étiquettes correspondantes.
    """
    all_data_list = []
    all_labels_list = []

    # Pour chaque élément on extrait les données et labels
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        data, labels = read_cifar_batch(filepath)

        all_data_list.append(data)
        all_labels_list.append(labels)
    
    all_data = np.concatenate(all_data_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    return all_data, all_labels

def split_dataset(data, labels, split: int ):
    """
    Divise les données et les labels en ensembles de training et de test.

    Paramètres :
        data (np.ndarray) : Les données à diviser.
        labels (np.ndarray) : Les labels associées.
        split (int) : Le pourcentage de données à allouer à l'ensemble de training.

    Retourne :
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] : Les données et labels de training et de test.
    """

    n_samples = data.shape[0]
    
    indices = np.random.permutation(n_samples)
    
    split_index = int(n_samples * split)
    
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    data_train = data[train_indices]
    labels_train = labels[train_indices]
    
    data_test = data[test_indices]
    labels_test = labels[test_indices]
    
    return data_train, labels_train, data_test, labels_test


if __name__ == "__main__":
    data, labels = read_cifar("data/cifar-10-batches-py")