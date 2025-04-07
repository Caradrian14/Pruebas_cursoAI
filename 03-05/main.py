import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
print(tf.__version__)

#a = tf.constant([[1,2],[3,4]])
#print(a)
a = tf.random.normal(shape=(10,3))
#print(a)

#print(a-a[0])
#print(tf.exp(a)[0].numpy())

# ---------Variables ---------
s = tf.Variable(tf.zeros_like(a[0]))
for i in a:
  s.assign_add(i)

print(s)

# ___Much better way to do it:___
tf.reduce_sum(a,axis=0)

# ----------Computing Gradients----------
# ----------Computing Gradients----------
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
  tape.watch(a)  # Start recording the history of operations applied to `a`
  c = tf.sqrt(tf.square(a) + tf.square(b))  # Do some math using `a`
  # What's the gradient of `c` with respect to `a`?
  dc_da = tape.gradient(c, a)
  print(dc_da)


#----Example 1: Linear Regression----

np.random.seed(13) # pick the seed for reproducability - change it to explore the effects of random variations

train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

plt.scatter(train_x,train_labels) #sale la grafica pero no se queda, usar debug