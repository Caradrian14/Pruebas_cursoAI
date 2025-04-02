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

# Computations on GPU -----------------------------
# Shuffle the data.
indices = np.random.permutation(len(train_x))
features = torch.tensor(train_x[indices],dtype=torch.float32)
labels = torch.tensor(train_labels[indices],dtype=torch.float32)

# --------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Doing computations on '+device)

### Changes here: indicate device
w = torch.tensor([100.0],requires_grad=True,dtype=torch.float32,device=device)
b = torch.zeros(size=(output_dim,),requires_grad=True,device=device)

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

batch_size = 4
for epoch in range(10):
  for i in range(0,len(features),batch_size):
    loss = train_on_batch(features[i:i+batch_size].view(-1,1),labels[i:i+batch_size])
  print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

np.random.seed(0) # pick the seed for reproducibility - change it to explore the effects of random variations

n = 100
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.1,class_sep=1.5)
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
  plt.show()
plot_dataset(train_x, train_labels)
#-----------------------------

# --------------Training One-Layer Perceptron---------------
# --------------NO FUNCIONAN_ERROR DE MULTIPLICACION DE MATRICES---------------
class Network():
  def __init__(self):
     self.W = torch.randn(size=(2,1),requires_grad=True)
     self.b = torch.zeros(size=(1,),requires_grad=True)

  def forward(self,x):
    return torch.matmul(x,self.W)+self.b

  def zero_grad(self):
    self.W.data.zero_()
    self.b.data.zero_()

  def update(self,lr=0.1):
    self.W.data.sub_(lr*self.W.grad)
    self.b.data.sub_(lr*self.b)

net = Network()

def train_on_batch(net, x, y):
  z = net.forward(x).flatten()
  loss = torch.nn.functional.binary_cross_entropy_with_logits(input=z,target=y)
  net.zero_grad()
  loss.backward()
  net.update()
  return loss

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

# Create a tf.data.Dataset object for easy batched iteration
dataset = torch.utils.data.TensorDataset(torch.tensor(train_x),torch.tensor(train_labels,dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset,batch_size=16)

list(dataloader)[0]

# for batch_idx, (data, labels) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     print("Data:", data)
#     print("Labels:", labels)
#     print("-" * 40)

for epoch in range(15):
  for (x, y) in dataloader:
    loss = train_on_batch(net,x,y)
  print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

  print(net.W, net.b)
plot_dataset(train_x,train_labels,net.W.detach().numpy(),net.b.detach().numpy())

pred = torch.sigmoid(net.forward(torch.tensor(valid_x)))
torch.mean(((pred.view(-1)>0.5)==(torch.tensor(valid_labels)>0.5)).type(torch.float32))
#-----------------------------

# --------------Neural Networks and Optimizers--------------
net = torch.nn.Linear(2,1) # 2 inputs, 1 output

print(list(net.parameters()))

optim = torch.optim.SGD(net.parameters(),lr=0.05)

val_x = torch.tensor(valid_x)
val_lab = torch.tensor(valid_labels)

for ep in range(10):
  for (x,y) in dataloader:
    z = net(x).flatten()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
  acc = ((torch.sigmoid(net(val_x).flatten())>0.5).float()==val_lab).float().mean()
  print(f"Epoch {ep}: last batch loss = {loss}, val acc = {acc}")


def train(net, dataloader, val_x, val_lab, epochs=10, lr=0.05):
  optim = torch.optim.Adam(net.parameters(),lr=lr)
  for ep in range(epochs):
    for (x,y) in dataloader:
      z = net(x).flatten()
      loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
      optim.zero_grad()
      loss.backward()
      optim.step()
    acc = ((torch.sigmoid(net(val_x).flatten())>0.5).float()==val_lab).float().mean()
    print(f"Epoch {ep}: last batch loss = {loss}, val acc = {acc}")

net = torch.nn.Linear(2,1)

train(net,dataloader,val_x,val_lab,lr=0.03)
#-----------------------------

# --------------Defining a Network as a Class --------------
