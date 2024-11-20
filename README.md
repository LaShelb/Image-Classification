
## Description

Ce projet a pour but de développer un programme de classification d'images en utilisant Python. Deux modèles de classification ont été implémentés : le k-plus proches voisins (KNN) et un réseau de neurones multicouche (MLP).

Le dataset CIFAR-10 a été utilisé, qui contient des images de 32x32 pixels classées en 10 catégories. Le projet consiste à lire les lots de données, les diviser en ensembles d'entraînement et de test, puis à prédire les étiquettes des images.

Le fichier read_cifar.py s'occupe de traiter les données d'entrée. Les fichiers knn.py et mlp.py contiennent respectivement les fonctions des k-plus proches voisins et du réseau de neurones multicouche. Enfin, le dossier results présente les résultats obtenus pour knn.py et mlp.py.


## Usage

1. **Cloner le dépôt :**

```bash
git clone https://gitlab.ec-lyon.fr/sramos/image-classification.git
```

2. **Installer les dépendances :**

```bash
pip install numpy matplotlib
```

3. **Télécharger les données :**

Télécharger le dataset CIFAR-10 et le mettre dans un dossier data.

4. **Exécuter le programme :**

```bash
python knn.py
python mlp.py
```