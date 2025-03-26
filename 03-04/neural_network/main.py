
# Imports
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.cm as cm
import pprint
import matplotlib.pyplot as plt
from Clases.Linear import Linear
from Clases.Softmax import Softmax
from Clases.CrossEntropyLoss import CrossEntropyLoss
from Clases.Neurona import Net
# Habilita el modo interactivo para ver graficas
plt.ion()


n = 100 # Numero de muestras
# Crea datos para la clasificación
# `make_classification` es una función de scikit-learn que crea un conjunto de datos aleatorio para pruebas de clasificación.

# Parámetros:
# - n_samples: Número de muestras (filas) en el conjunto de datos. En este caso, 100.
# - n_features: Número de características (columnas) en el conjunto de datos. Aquí, 2.
# - n_redundant: Número de características redundantes. En este caso, 0, lo que significa que no hay características redundantes.
# - n_informative: Número de características informativas. Aquí, 2, lo que significa que ambas características son informativas.
# - flip_y: Proporción de etiquetas que se invierten aleatoriamente. Aquí, 0.2, lo que significa que el 20% de las etiquetas se invierten.
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.2)

# Convertir las características (X) e (Y) a tipo de dato float32 para que no pese tanto en el algoritmo de despues
X = X.astype(np.float32)
Y = Y.astype(np.int32)

# Dividir el conjunto de datos en subconjuntos de entrenamiento y prueba
# `np.split` es una función de NumPy que divide un array en múltiples subarrays.

# Parámetros:
# - `X`: El array de características que se va a dividir.
# - `[n*8//10]`: Lista de índices donde se realizarán las divisiones. Aquí, se divide en el índice que corresponde al 80% del tamaño total de `X`.
#   Esto significa que el 80% de los datos se usarán para entrenamiento y el 20% restante para pruebas.
train_x, test_x = np.split(X, [n * 8 // 10])

# Dividir las etiquetas en subconjuntos de entrenamiento y prueba
# - `Y`: El array de etiquetas que se va a dividir.
# - `[n*8//10]`: Lista de índices donde se realizarán las divisiones. Aquí, se divide en el índice que corresponde al 80% del tamaño total de `Y`.
#   Esto significa que el 80% de las etiquetas se usarán para entrenamiento y el 20% restante para pruebas.
train_labels, test_labels = np.split(Y, [n * 8 // 10])


def train_epoch(net, train_x, train_labels, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    """
       Entrena una red neuronal durante una época (un lote de datos) completa.

       Parámetros:
       - net: Instancia de la red neuronal que se va a entrenar. Debe tener métodos `forward`, `backward` y `update`.
       - train_x: Array de características de entrenamiento.
       - train_labels: Array de etiquetas de entrenamiento.
       - loss: Instancia de la función de pérdida a utilizar. Por defecto, es una instancia de `CrossEntropyLoss`.
       - batch_size: Tamaño del lote (batch) de datos a procesar en cada iteración. Por defecto, es 4.
       - lr: Tasa de aprendizaje para actualizar los parámetros de la red. Por defecto, es 0.1.

       La función realiza las siguientes operaciones:
       1. Divide los datos de entrenamiento en lotes.
       2. Para cada lote, realiza una pasada hacia adelante (forward pass) para obtener las predicciones.
       3. Calcula la pérdida utilizando la función de pérdida especificada.
       4. Realiza una pasada hacia atrás (backward pass) para calcular los gradientes.
       5. Actualiza los parámetros de la red utilizando la tasa de aprendizaje especificada.
       """
    for i in range(0, len(train_x), batch_size):
        # Seleccionar un lote de datos de entrenamiento
        xb = train_x[i:i + batch_size]
        yb = train_labels[i:i + batch_size]

        # Pasada hacia adelante: obtener las predicciones de la red
        p = net.forward(xb)
        # Calcular la pérdida para el lote actual
        l = loss.forward(p, yb)
        # Pasada hacia atrás: calcular los gradientes de la pérdida
        dp = loss.backward(l)
        # Propagar los gradientes hacia atrás a través de la red
        dx = net.backward(dp)
        # Actualizar los parámetros de la red utilizando la tasa de aprendizaje
        net.update(lr)

def get_loss_acc(x, y, loss=CrossEntropyLoss()):
    """
    Calcula la pérdida y la precisión de un modelo de red neuronal para un conjunto de datos dado.

    Parámetros:
    - x: Array de características de entrada.
    - y: Array de etiquetas verdaderas correspondientes a las características de entrada.
    - loss: Instancia de la función de pérdida a utilizar. Por defecto, es una instancia de `CrossEntropyLoss`.

    Retorna:
    - l: Valor de la pérdida calculada para el conjunto de datos.
    - acc: Precisión del modelo, definida como la proporción de predicciones correctas.
    """
    # Realizar una pasada hacia adelante a través de la red para obtener las predicciones
    p = net.forward(x)

    # Calcular la pérdida utilizando la función de pérdida especificada
    l = loss.forward(p, y)

    # Obtener las predicciones de clase tomando el índice con la mayor probabilidad
    pred = np.argmax(p, axis=1)

    # Calcular la precisión como la proporción de predicciones correctas
    acc = (pred == y).mean()

    # Retornar la pérdida y la precisión
    return l, acc

import matplotlib.pyplot as plt

def train_and_plot(n_epoch, net, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    """
    Entrena una red neuronal durante un número especificado de épocas y grafica el progreso del entrenamiento.

    Parámetros:
    - n_epoch: Número de épocas para entrenar la red.
    - net: Instancia de la red neuronal que se va a entrenar.
    - loss: Instancia de la función de pérdida a utilizar. Por defecto, es una instancia de `CrossEntropyLoss`.
    - batch_size: Tamaño del lote (batch) de datos a procesar en cada iteración. Por defecto, es 4.
    - lr: Tasa de aprendizaje para actualizar los parámetros de la red. Por defecto, es 0.1.

    Retorna:
    - train_acc: Array con la precisión de entrenamiento para cada época.
    - valid_acc: Array con la precisión de validación para cada época.
    """
    # Configurar la figura y los ejes para los gráficos
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlim(0, n_epoch + 1)
    ax[0].set_ylim(0, 1)

    # Inicializar arrays para almacenar la precisión de entrenamiento y validación
    train_acc = np.empty((n_epoch, 3))
    train_acc[:] = np.nan
    valid_acc = np.empty((n_epoch, 3))
    valid_acc[:] = np.nan

    # Iterar sobre el número de épocas
    for epoch in range(1, n_epoch + 1):
        # Entrenar la red durante una época
        train_epoch(net, train_x, train_labels, loss, batch_size, lr)

        # Calcular la pérdida y la precisión en el conjunto de entrenamiento
        tloss, taccuracy = get_loss_acc(train_x, train_labels, loss)
        train_acc[epoch - 1, :] = [epoch, tloss, taccuracy]

        # Calcular la pérdida y la precisión en el conjunto de validación
        vloss, vaccuracy = get_loss_acc(test_x, test_labels, loss)
        valid_acc[epoch - 1, :] = [epoch, vloss, vaccuracy]

        # Ajustar los límites del eje y para la precisión
        ax[0].set_ylim(0, max(max(train_acc[:, 2]), max(valid_acc[:, 2])) * 1.1)

        # Graficar el progreso del entrenamiento
        plot_training_progress(train_acc[:, 0], (train_acc[:, 2], valid_acc[:, 2]), fig, ax[0])

        # Graficar la frontera de decisión
        plot_decision_boundary(net, fig, ax[1])

        # Actualizar la figura
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Retornar la precisión de entrenamiento y validación
    return train_acc, valid_acc

def plot_training_progress(x, y_data, fig, ax):
    """
    Grafica el progreso del entrenamiento y la validación a lo largo de las épocas.

    Parámetros:
    - x: Array de valores para el eje x (generalmente, el número de épocas).
    - y_data: Tupla de arrays de valores para el eje y (precisión de entrenamiento y validación).
    - fig: Figura de matplotlib en la que se dibujará el gráfico.
    - ax: Ejes de matplotlib en los que se dibujará el gráfico.

    La función actualiza el gráfico con las nuevas métricas de entrenamiento y validación.
    """
    # Estilos de línea para diferenciar entre la precisión de entrenamiento y validación
    styles = ['k--', 'g-']

    # Eliminar las líneas anteriores del gráfico
    while ax.lines:
        line = ax.lines[0]
        line.remove()

    # Dibujar las líneas actualizadas para la precisión de entrenamiento y validación
    for i in range(len(y_data)):
        ax.plot(x, y_data[i], styles[i])

    # Añadir una leyenda al gráfico
    ax.legend(ax.lines, ['training accuracy', 'validation accuracy'],
              loc='upper center', ncol=2)

def plot_decision_boundary(net, fig, ax):
    """
    Grafica la frontera de decisión de una red neuronal en un espacio de características bidimensional.

    Parámetros:
    - net: Instancia de la red neuronal que se utilizará para hacer predicciones.
    - fig: Figura de matplotlib en la que se dibujará el gráfico.
    - ax: Ejes de matplotlib en los que se dibujará el gráfico.

    La función genera un gráfico de contorno que muestra las regiones de decisión de la red neuronal
    y los puntos de datos de entrenamiento.
    """
    draw_colorbar = True

    # Eliminar gráficos anteriores
    while ax.collections:
        line = ax.collections[0]
        line.remove()
        draw_colorbar = False

    # Generar una cuadrícula para el gráfico a mostrar
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel().astype('float32'), yy.ravel().astype('float32')]

    # Asegurarse de que la cuadrícula tenga la misma dimensionalidad que los datos de entrenamiento
    n_classes = max(train_labels) + 1
    while train_x.shape[1] > grid_points.shape[1]:
        grid_points = np.c_[grid_points,
                            np.empty(len(xx.ravel())).astype('float32')]
        grid_points[:, -1].fill(train_x[:, grid_points.shape[1] - 1].mean())

    # Evaluar las predicciones de la red en los puntos de la cuadrícula
    prediction = np.array(net.forward(grid_points))

    # Calcular los valores Z para el gráfico de contorno
    if n_classes == 2:
        Z = np.array([0.5 + (p[0] - p[1]) / 2.0 for p in prediction]).reshape(xx.shape)
    else:
        Z = np.array([p.argsort()[-1] / float(n_classes - 1) for p in prediction]).reshape(xx.shape)

    # Dibujar el gráfico de contorno
    levels = np.linspace(0, 1, 40)
    cs = ax.contourf(xx, yy, Z, alpha=0.4, levels=levels)

    # Añadir una barra de color si es necesario
    if draw_colorbar:
        fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1])

    # Definir el mapa de colores para las clases
    c_map = [cm.jet(x) for x in np.linspace(0.0, 1.0, n_classes)]
    colors = [c_map[l] for l in train_labels]

    # Graficar los puntos de datos de entrenamiento
    ax.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=colors, s=60, alpha=0.5)



net = Net()

print("Initial loss={}, accuracy={}: ".format(*get_loss_acc(train_x, train_labels)))
train_epoch(net, train_x, train_labels)
print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_x, train_labels)))
print("Test loss={}, accuracy={}: ".format(*get_loss_acc(test_x, test_labels)))

net.add(Linear(2,2))
net.add(Softmax())

res = train_and_plot(30,net,lr=0.005)
plt.show(block=True)