import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5


#--------Computational Graph and GPU Computations--------
#Tensorflow allows us to mark our Python function using @tf.function decorator, which will make this function a part of the same computational graph.
# This decorator can be applied to functions that use standard Tensorflow tensor operations.

# loss function:
input_dim = 1
output_dim = 1
learning_rate = 0.1

# This is our weight matrix
w = tf.Variable([[100.0]])
# This is our bias vector
b = tf.Variable(tf.zeros(shape=(output_dim,)))

def f(x):
  return tf.matmul(x,w) + b

def compute_loss(labels, predictions):
  return tf.reduce_mean(tf.square(labels - predictions))
# We will train the model on a series of minibatches. We will use gradient descent, adjusting model parameters using the following formulae
def train_on_batch(x, y):
  with tf.GradientTape() as tape:
    predictions = f(x)
    loss = compute_loss(y, predictions)
    # Note that `tape.gradient` works with a list as well (w, b).
    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
  w.assign_sub(learning_rate * dloss_dw)
  b.assign_sub(learning_rate * dloss_db)
  return loss

@tf.function
def train_on_batch(x, y):
  with tf.GradientTape() as tape:
    predictions = f(x)
    loss = compute_loss(y, predictions)
    # Note that `tape.gradient` works with a list as well (w, b).
    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
  w.assign_sub(learning_rate * dloss_dw)
  b.assign_sub(learning_rate * dloss_db)
  return loss
# The code has not changed, but if you were running this code on GPU and on larger dataset - you would have noticed the difference in speed.

# --------------Dataset API --------------
# Tensorflow contains a convenient API to work with data. Let's try to use it. We will also train our model from scratch.
w.assign([[10.0]])
b.assign([0.0])

# Create a tf.data.Dataset object for easy batched iteration
dataset = tf.data.Dataset.from_tensor_slices((train_x.astype(np.float32), train_labels.astype(np.float32)))
dataset = dataset.shuffle(buffer_size=1024).batch(256)

for epoch in range(10):
  for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(tf.reshape(x,(-1,1)), tf.reshape(y,(-1,1)))
  print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

# ----Example 2: Classification----
# The core model is similar to regression, but we need to use different loss function. Let's start by generating sample data:

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

plot_dataset(train_x, train_labels)