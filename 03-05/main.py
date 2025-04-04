# Keras is a part of Tensorflow 2.x framework. Let's make sure we have version 2.x.x of Tensorflow installed:
# No funciona con python 3.13 que es el que estoy usando, hay que usar el python 3.8 o 3.9
# cambiar el
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

print(f'Tensorflow version = {tf.__version__}')
print(f'Keras version = Keras NO HAY METODO PARA VER VERSIONES')
# se ha mirado desde la terminal , la version es la 2.31.1
