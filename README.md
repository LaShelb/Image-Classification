# Image Classification with KNN and MLP

This project demonstrates how to implement **image classification** using two different approaches in Python: **k-Nearest Neighbors (k-NN)** and a **Multilayer Perceptron (MLP)** neural network. It utilizes the popular **CIFAR-10** dataset, which contains 32×32 color images categorized into 10 classes.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Contact](#contact)  

---

## Project Overview

The goal of this project is to explore two fundamental machine learning methods for image classification:

1. **k-Nearest Neighbors (k-NN)**
2. **Multilayer Perceptron (MLP)**

By leveraging these approaches, we can gain an understanding of how classic machine learning algorithms (like k-NN) compare with simpler neural networks (like an MLP) in terms of performance, accuracy, and computational cost.

**Key highlights:**
- Preprocessing the CIFAR-10 dataset using a custom Python script.
- Implementing both k-NN and MLP from scratch (using NumPy for matrix operations).
- Comparing performance metrics on the CIFAR-10 test set.

---

## Dataset

- **CIFAR-10**: The dataset consists of 60,000 32×32 color images, classified into 10 categories such as airplanes, cars, birds, cats, deers, dogs, frogs, horses, ships, and trucks.

You can find more information about the dataset and download it from the [official CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html). After downloading, place the data in a folder named `data/` at the root of this project.

---

## Project Structure

```bash
.
├── data/                  # (Not included in the repository) CIFAR-10 data folder
├── mlp.py                 # MLP-based classification
├── knn.py                 # k-NN-based classification
├── read_cifar.py          # Utility script for reading/preprocessing CIFAR-10
├── results/
│   ├── knn_results.txt    # Example output/results for k-NN
│   └── mlp_results.txt    # Example output/results for MLP
├── requirements.txt       # List of dependencies (optional)
└── README.md              # Project documentation
```


- **`read_cifar.py`** handles data loading and preprocessing.  
- **`knn.py`** contains the implementation of the k-NN classifier.  
- **`mlp.py`** contains the implementation of the MLP classifier.  
- **`results/`** directory stores any generated outputs or performance metrics.  

---

## Installation

1. **Clone the repository**:
```bash
git clone https://gitlab.ec-lyon.fr/sramos/image-classification.git
cd image-classification
```
2. **Install dependencies (ideally in a virtual environment)**:
```bash
pip install numpy matplotlib
```

Or install from the provided requirements.txt if available:
```bash
pip install -r requirements.txt
```
3. **Download the CIFAR-10 dataset** and place it in the data/ directory at the root of this project.

## Usage

Once you have the dataset and dependencies ready, you can run:

1. **k-NN classification**:
```bash
python knn.py
```
This will load the CIFAR-10 dataset and run the k-NN classifier, outputting the classification accuracy and possibly other metrics.

2. **MLP classification**:
```bash
python mlp.py
```
This will similarly load the dataset, train the MLP model, and display the results.


You can adjust hyperparameters (e.g., the number of neighbors in k-NN, or the number of hidden layers/neurons in MLP) directly in the respective Python scripts.

## Results

- **k-NN**:
    - Typically straightforward to implement but may be slow for large datasets due to the computation of distances for each query.
    - The results stored in results/knn_results.txt show the accuracy obtained after classification on the CIFAR-10 test set.


- **MLP**:
    - More computationally intensive during training but efficient during inference.
    - The results stored in results/mlp_results.txt demonstrate the accuracy after training and testing on CIFAR-10.

---

## Contact

For any questions or collaborations, feel free to reach out via [LinkedIn](www.linkedin.com/in/simon-ramos-190064234).