import numpy as np

"""
Esta clase es un componente 
básico de las redes neuronales, permitiendo la 
transformación de datos de entrada 
a través de pesos y sesgos ajustables.
"""
class Linear:
    """
       Clase que representa una capa lineal (fully connected) en una red neuronal.
       Esta capa realiza una transformación lineal de la entrada utilizando pesos y sesgos.
       """
    def __init__(self, nin, nout):
        """
        Inicializa una capa lineal con pesos y sesgos.

        Parámetros:
        - nin: Número de características de entrada (número de columnas en la matriz de pesos).
        - nout: Número de características de salida (número de filas en la matriz de pesos).

        Atributos:
        - W: Matriz de pesos de tamaño (nout, nin), inicializada con valores aleatorios de una distribución normal.
        - b: Vector de sesgos de tamaño (1, nout), inicializado con ceros.
        - dW: Gradientes de los pesos, inicializados con ceros.
        - db: Gradientes de los sesgos, inicializados con ceros.
        """
        self.W = np.random.normal(0, 1.0 / np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1, nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        Realiza una pasada hacia adelante a través de la capa lineal.

        Parámetros:
        - x: Entrada a la capa. Debe ser un array de tamaño (batch_size, nin).

        Retorna:
        - La salida de la capa después de aplicar la transformación lineal.
        """
        self.x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, dz):
        """
        Realiza una pasada hacia atrás a través de la capa lineal para calcular los gradientes.

        Parámetros:
        - dz: Gradiente de la pérdida con respecto a la salida de la capa.

        Retorna:
        - dx: Gradiente de la pérdida con respecto a la entrada de la capa.
        """
        # Calcular el gradiente de la pérdida con respecto a la entrada
        dx = np.dot(dz, self.W)
        # Calcular el gradiente de la pérdida con respecto a los pesos y sesgos
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)
        # Almacenar los gradientes calculados
        self.dW = dW
        self.db = db

        return dx

    def update(self, lr):
        """
        Actualiza los pesos y sesgos de la capa utilizando los gradientes y la tasa de aprendizaje.

        Parámetros:
        - lr: Tasa de aprendizaje para actualizar los pesos y sesgos.
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db