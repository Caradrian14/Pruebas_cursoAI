# Introduction to Tensorflow and Keras

## Basic Concepts: Tensor

Tensor is a multi-dimensional array. It is very convenient to use tensors to represent different types of data:

- 400x400 - black-and-white picture
- 400x400x3 - color picture
- 16x400x400x3 - minibatch of 16 color pictures
- 25x400x400x3 - one second of 25-fps video
- 8x25x400x400x3 - minibatch of 8 1-second videos

### Simple Tensors

You can easily create simple tensors from lists of np-arrays, or generate random ones

````
a = tf.constant([[1,2],[3,4]])
print(a)
a = tf.random.normal(shape=(10,3))
print(a)
````

You can use arithmetic operations on tensors, which are performed element-wise, as in numpy. Tensors are automatically expanded to required dimension, if needed. To extract numpy-array from tensor, use .numpy():

### Variables

Variables are useful to represent tensor values that can be modified using assign and assign_add. They are often used to represent neural network weights.