# Keras is a part of Tensorflow 2.x framework. Let's make sure we have version 2.x.x of Tensorflow installed:
# No funciona con python 3.13 que es el que estoy usando, hay que usar el python 3.8 o 3.9
# cambiar el
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

np.random.seed(0) # pick the seed for reproducibility - change it to explore the effects of random variations

n = 100
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.05,class_sep=1.5)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

split = [ 70*n//100 ] # crea una lista doonde solo se guarda el 70% de n
train_x, test_x = np.split(X, split)
train_labels, test_labels = np.split(Y, split)

train_x_norm = (train_x-np.min(train_x,axis=0)) / (np.max(train_x,axis=0)-np.min(train_x,axis=0))
test_x_norm = (test_x-np.min(train_x,axis=0)) / (np.max(train_x,axis=0)-np.min(train_x,axis=0))
