# Simplest Introduction to Neural Networks with Keras
There are several frameworks for training neural networks. However, if you want to get started fast and not go into much detail on how things work internally - you should consider using Keras. This short tutorial will get you started, and if you want to get deeper into understanding how things work - look into Introduction to Tensorflow and Keras notebook.

## Recordar
La libreria de Keras usa tensorflow, pare que funcione bien es necesario usar una version de python compatible, puesto que a partir de la 3.9 y aparece que da problemas

## Getting things ready
`pip install tensorflow`

Codigo simple de inicio:

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
print(f'Tensorflow version = {tf.__version__}')
# cuidado que parece que no hay metodo de version
print(f'Keras version = {keras.__version__}')
```

# Basic Concepts: Tensor
Basicamente una matriz
Tensor is a multi-dimensional array. It is very convenient to use tensors to represent different types of data.

Tensors give us a convenient way to represent input/output data, as well we weights inside the neural network.


# Normalizing Data

Before training, it is common to bring our input features to the standard range of [0,1] (or [-1,1]). The exact reasons for that we will discuss later in the course, but in short the reason is the following. We want to avoid values that flow through our network getting too big or too small, and we normally agree to keep all values in the small range close to 0. Thus we initialize the weights with small random numbers, and we keep signals in the same range.

When normalizing data, we need to subtract min value and divide by range. We compute min value and range using training data, and then normalize test/validation dataset using the same min/range values from the training set. This is because in real life we will only know the training set, and not all incoming new values that the network would be asked to predict. Occasionally, the new value may fall out of the [0,1] range, but that's not crucial.

# Training One-Layer Network (Perceptron)

In many cases, a neural network would be a sequence of layers. It can be defined in Keras using Sequential model in the following manner:
````
model = keras.models.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation(keras.activations.sigmoid))

model.summary()
````
Here, we first create the model, and then add layers to it:

- First Input layer (which is not strictly speaking a layer) contains the specification of network's input size.
- Dense layer is the actual perceptron that contains trainable weights
- Finally, there is a layer with sigmoid Activation function to bring the result of the network into 0-1 range (to make it a probability).
Input size, as well as activation function, can also be specified directly in the Dense layer for brevity:
````
model = keras.models.Sequential()
model.add(keras.layers.Dense(1,input_shape=(2,),activation='sigmoid'))
model.summary()
````
Before training the model, we need to compile it, which essentially mean specifying:
- Loss function, which defines how loss is calculated. Because we have two-class classification problem, we will use binary cross-entropy loss.
- Optimizer to use. The simplest option would be to use sgd for stochastic gradient descent, or you can use more sophisticated optimizers such as adam.
- Metrics that we want to use to measure success of our training. Since it is classification task, a good metrics would be Accuracy (or acc for short).
We can specify loss, metrics and optimizer either as strings, or by providing some objects from Keras framework. In our example, we need to specify learning_rate parameter to fine-tune learning speed of our model, and thus we provide full name of Keras SGD optimizer.

````
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2),loss='binary_crossentropy',metrics=['acc'])
````

After compiling the model, we can do the actual training by calling fit method. The most important parameters are:

- x and y specify training data, features and labels respectively
- If we want validation to be performed on each epoch, we can specify validation_data parameter, which would be a tuple of features and labels
- epochs specified the number of epochs
- If we want training to happen in minibatches, we can specify batch_size parameter. You can also pre-batch the data manually before passing it to x/y/validation_data, in which case you do not need batch_size

````
model.fit(x=train_x_norm,y=train_labels,validation_data=(test_x_norm,test_labels),epochs=10,batch_size=1)
````