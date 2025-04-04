# Keras is a part of Tensorflow 2.x framework. Let's make sure we have version 2.x.x of Tensorflow installed:
# No funciona con python 3.13 que es el que estoy usando, hay que usar el python 3.8 o 3.9
# cambiar el
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


# Normalizing Data

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

# Training One-Layer Network (Perceptron)
model = keras.models.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation(keras.activations.sigmoid))
#model.summary() # da un resumen en terminal

model = keras.models.Sequential()
model.add(keras.layers.Dense(1,input_shape=(2,),activation='sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2),loss='binary_crossentropy',metrics=['acc'])

model.fit(x=train_x_norm,y=train_labels,validation_data=(test_x_norm,test_labels),epochs=10,batch_size=1)

def plot_dataset(features, labels, W=None, b=None):
  # prepare the plot
  fig, ax = plt.subplots(1, 1)
  ax.set_xlabel('$x_i[0]$ -- (feature 1)')
  ax.set_ylabel('$x_i[1]$ -- (feature 2)')
  colors = ['r' if l else 'b' for l in labels]
  ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha=0.5)
  if W is not None:
    min_x = min(features[:, 0])
    max_x = max(features[:, 1])
    min_y = min(features[:, 1]) * (1 - .1)
    max_y = max(features[:, 1]) * (1 + .1)
    cx = np.array([min_x, max_x], dtype=np.float32)
    cy = (0.5 - W[0] * cx - b) / W[1]
    ax.plot(cx, cy, 'g')
    ax.set_ylim(min_y, max_y)
  plt.show()

plot_dataset(train_x,train_labels,model.layers[0].weights[0],model.layers[0].weights[1])