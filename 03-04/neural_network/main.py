
# Imports
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.cm as cm
import pprint
import matplotlib.pyplot as plt

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

from Clases.CrossEntropyLoss import CrossEntropyLoss

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

from Clases.CrossEntropyLoss import CrossEntropyLoss
