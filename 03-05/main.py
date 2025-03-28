import torch

# Crear un tensor de 2x2 con valores específicos
a = torch.tensor([[1,2],[3,4]])
#print(a)
# Reasignar `a` a un tensor de tamaño 10x3 con valores aleatorios de una distribución normal
a = torch.randn(size=(10,3))
#print(a)

# Restar la primera fila del tensor `a` de todas las filas
#print(a-a[0])

# Calcular la exponencial de cada elemento del tensor `a` y convertir la primera fila a un array de NumPy
#print(torch.exp(a)[0].numpy())

#----------
# Crear un tensor escalar con el valor 5
u = torch.tensor(5)
# Sumar 3 al tensor `u` sin modificar `u` en su lugar (out-of-place)
print("Result when adding out-of-place:",u.add(torch.tensor(3)))
# Sumar 3 al tensor `u` modificando `u` en su lugar (in-place)
u.add_(torch.tensor(3))
print("Result after adding in-place:", u)

#----------
# Crear un tensor de ceros con la misma forma que la primera fila de `a`
s = torch.zeros_like(a[0])

# Sumar todas las filas de `a` al tensor `s`
for i in a:
  s.add_(i) # Suma cada fila de `a` a `s` en su lugar

print(s)

# Alternativamente, sumar todas las filas de `a` a lo largo del eje 0 (columnas)
# Esto es más conveniente y eficiente que el bucle anterior
torch.sum(a,axis=0)
#----------
# Computing Gradients
print("Computing Gradients")
# Crear un tensor `a` de tamaño 2x2 con valores aleatorios de una distribución normal
# `requires_grad=True` indica que queremos calcular los gradientes con respecto a este tensor
a = torch.randn(size=(2, 2), requires_grad=True)
# Crear un tensor `b` de tamaño 2x2 con valores aleatorios de una distribución normal
# No se requieren gradientes para `b`
b = torch.randn(size=(2, 2))

# Realizar algunas operaciones matemáticas usando `a` y `b`
# Calcular la media de la raíz cuadrada de la suma de los cuadrados de `a` y `b`
c = torch.mean(torch.sqrt(torch.square(a) + torch.square(b)))
# Llamar a `backward()` en `c` para calcular los gradientes de `c` con respecto a `a`
c.backward()
# What's the gradient of `c` with respect to `a`?
print(a.grad)

#------------
print("Calculos con c")
# Realizar algunas operaciones matemáticas usando `a` y `b`
# Calcular la media de la raíz cuadrada de la suma de los cuadrados de `a` y `b`
c = torch.mean(torch.sqrt(torch.square(a) + torch.square(b)))

# Llamar a `backward()` en `c` para calcular los gradientes de `c` con respecto a `a`
# `retain_graph=True` permite llamar a `backward()` múltiples veces en el mismo gráfico de computación
c.backward(retain_graph=True)
# Llamar a `backward()` nuevamente
# Los gradientes se acumularán en `a.grad`
c.backward(retain_graph=True)
print(a.grad)

# Reiniciar los gradientes acumulados en `a.grad` a cero
a.grad.zero_()
# Llamar a `backward()` nuevamente después de reiniciar los gradientes
c.backward()
print(a.grad)
print("Tensor C")
print(c)

# -----
print("Matriz Jacobina")
# Realizar algunas operaciones matemáticas usando `a` y `b`
# Calcular la raíz cuadrada de la suma de los cuadrados de `a` y `b`
c = torch.sqrt(torch.square(a) + torch.square(b))
# Llamar a `backward()` en `c` para calcular los gradientes de `c` con respecto a `a`
# `torch.eye(2)` es una matriz identidad de 2x2, que se utiliza para inicializar el gradiente de `c`
# Esto es equivalente a calcular la matriz Jacobiana de `c` con respecto a `a`
c.backward(torch.eye(2)) # eye(2) means 2x2 identity matrix
# Imprimir los gradientes calculados
# `a.grad` contiene las derivadas parciales de `c` con respecto a cada elemento de `a`
# Estos gradientes representan cómo cambia `c` con respecto a pequeños cambios en `a`
print(a.grad)

# Relación con la Matriz Jacobiana:
# En este contexto, la matriz Jacobiana es una matriz de derivadas parciales que describe cómo cambia `c`
# con respecto a `a`. Al utilizar `torch.eye(2)`, estamos efectivamente calculando una fila de la matriz
# Jacobiana, que representa la derivada de `c` con respecto a cada elemento de `a`.
#
# En redes neuronales, la matriz Jacobiana es fundamental para la retropropagación, ya que permite calcular
# cómo cambian las salidas de la red con respecto a los parámetros del modelo. Esto es crucial para actualizar
# los parámetros del modelo de manera eficiente durante el entrenamiento.

# -------------------
print("Optimization Using Gradient Descent")
# Crear un tensor `x` de tamaño 2 con valores iniciales de cero
# `requires_grad=True` indica que queremos calcular los gradientes con respecto a este tensor
x = torch.zeros(2, requires_grad=True)

# Definir una función `f` que calcula la suma de los cuadrados de las diferencias
# entre `x` y el tensor [3, -2]. Esto es una función de pérdida cuadrática simple.
f = lambda x: (x - torch.tensor([3, -2])).pow(2).sum()

# Definir la tasa de aprendizaje (learning rate)
lr = 0.1

# Realizar 15 iteraciones de descenso de gradiente
for i in range(15):
    # Calcular el valor de la función de pérdida `f` en el punto `x`
    y = f(x)

    # Calcular los gradientes de `y` con respecto a `x`
    y.backward()

    # Obtener los gradientes calculados
    gr = x.grad

    # Actualizar los valores de `x` usando la tasa de aprendizaje y los gradientes
    # `x.data.add_(-lr * gr)` actualiza `x` en su lugar, restando el producto de la tasa de aprendizaje y los gradientes
    x.data.add_(-lr * gr)

    # Reiniciar los gradientes a cero para la siguiente iteración
    x.grad.zero_()

    # Imprimir los valores actualizados de `x` en cada iteración
    print("Step {}: x[0]={}, x[1]={}".format(i, x[0], x[1]))

#-----------------
# Regresiones Liniales
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

# Establecer una semilla para la reproducibilidad de los resultados aleatorios
np.random.seed(13)

# Crear un conjunto de datos de entrenamiento
# `train_x` es un array de 120 puntos equiespaciados entre 0 y 3
train_x = np.linspace(0, 3, 120)

# `train_labels` es una línea recta con ruido gaussiano añadido
# La relación es y = 2x + 0.9 + ruido
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

# Crear un gráfico de dispersión de los datos de entrenamiento
plt.scatter(train_x, train_labels, color='blue', alpha=0.6, label='Datos de entrenamiento')

# Añadir etiquetas y título al gráfico
plt.title('Gráfico de Dispersión de Datos de Entrenamiento')
plt.xlabel('train_x')
plt.ylabel('train_labels')
# Añadir una leyenda
plt.legend()
# Mostrar el gráfico
plt.show()

#--------------
print("Funciones de perdida")
input_dim = 1
output_dim = 1
learning_rate = 0.1

# This is our weight matrix
w = torch.tensor([100.0],requires_grad=True,dtype=torch.float32)
# This is our bias vector
b = torch.zeros(size=(output_dim,),requires_grad=True)

def f(x):
  return torch.matmul(x,w) + b

def compute_loss(labels, predictions):
  return torch.mean(torch.square(labels - predictions))

def train_on_batch(x, y):
  predictions = f(x)
  loss = compute_loss(y, predictions)
  loss.backward()
  w.data.sub_(learning_rate * w.grad)
  b.data.sub_(learning_rate * b.grad)
  w.grad.zero_()
  b.grad.zero_()
  return loss

# -------------- Codigo funcional de la funcion de perdida --------------

# This is our weight matrix
w = torch.tensor([100.0],requires_grad=True,dtype=torch.float32)
# This is our bias vector
b = torch.zeros(size=(output_dim,),requires_grad=True)

train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

def f(x):
  return torch.matmul(x,w) + b

def compute_loss(labels, predictions):
  return torch.mean(torch.square(labels - predictions))

def train_on_batch(x, y):
  predictions = f(x)
  loss = compute_loss(y, predictions)
  loss.backward()
  w.data.sub_(learning_rate * w.grad)
  b.data.sub_(learning_rate * b.grad)
  w.grad.zero_()
  b.grad.zero_()
  return loss

# Shuffle the data.
indices = np.random.permutation(len(train_x))
features = torch.tensor(train_x[indices],dtype=torch.float32)
labels = torch.tensor(train_labels[indices],dtype=torch.float32)

batch_size = 4
for epoch in range(10):
  for i in range(0,len(features),batch_size):
    loss = train_on_batch(features[i:i+batch_size].view(-1,1),labels[i:i+batch_size])
  print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

plt.scatter(train_x,train_labels)
x = np.array([min(train_x),max(train_x)])
with torch.no_grad():
  y = w.numpy()*x+b.numpy()
plt.plot(x,y,color='red')
# Añadir etiquetas y leyenda
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Regresión Lineal')

# Mostrar la gráfica
plt.show()
# ----------------------------- --------------

# Computations on GPU
