import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5
np.random.seed(0) # pick the seed for reproducibility - change it to explore the effects of random variations

n = 100
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.05,class_sep=1.5)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

split = [ 70*n//100, (15+70)*n//100 ]
train_x, valid_x, test_x = np.split(X, split)
train_labels, valid_labels, test_labels = np.split(Y, split)

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
  fig.show()
train_x_norm = (train_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))
valid_x_norm = (valid_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))
test_x_norm = (test_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))

#---------------Keras---------------
#-----Deep Learning for Humans-----
inputs = tf.keras.Input(shape=(2,))
z = tf.keras.layers.Dense(1,kernel_initializer='glorot_uniform',activation='sigmoid')(inputs)
model = tf.keras.models.Model(inputs,z)

model.compile(tf.keras.optimizers.Adam(0.1),'binary_crossentropy',['accuracy'])
model.summary()
h = model.fit(train_x_norm,train_labels,batch_size=8,epochs=15)

plt.plot(h.history['accuracy'])

# -------Sequential API-------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5,activation='sigmoid',input_shape=(2,)))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(tf.keras.optimizers.Adam(0.1),'binary_crossentropy',['accuracy'])
model.summary()
model.fit(train_x_norm,train_labels,validation_data=(test_x_norm,test_labels),batch_size=8,epochs=15)