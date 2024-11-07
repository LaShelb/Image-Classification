import os
import pickle
import numpy as np

def read_cifar_batch(path_batch: str):

    with open(path_batch, 'rb') as f:

        batch = pickle.load(f, encoding='bytes')
        
    data = batch[b'data']
    
    labels = batch[b'labels']
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    return data, labels

def read_cifar(directory_path: str):
    all_data_list = []
    all_labels_list = []

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        data, labels = read_cifar_batch(filepath)

        all_data_list.append(data)
        all_labels_list.append(labels)
    
    all_data = np.concatenate(all_data_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    return all_data, all_labels

def split_dataset(data, labels, split: int ):

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