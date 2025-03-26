
"""
Esta clase es fundamental para construir y
entrenar redes neuronales,
permitiendo la flexibilidad de agregar
diferentes tipos de capas y realizar el entrenamiento mediante
propagación hacia adelante y hacia atrás.
"""

class Net:
    """
    Clase que representa una red neuronal.
    Esta clase permite agregar capas a la red, realizar pasadas hacia adelante y hacia atrás,
    y actualizar los parámetros de las capas.
    """
    def __init__(self):
        """
        Inicializa una instancia de la red neuronal.
        - `layers`: Lista que almacenará las capas de la red.
        """
        self.layers = []

    def add(self, l):
        """
        Agrega una capa a la red.

        Parámetros:
        - l: Instancia de una capa que se agregará a la red.
        """
        self.layers.append(l)

    def forward(self, x):
        """
        Realiza una pasada hacia adelante a través de todas las capas de la red.

        Parámetros:
        - x: Entrada a la red. Puede ser un array de características.

        Retorna:
        - La salida de la red después de pasar por todas las capas.
        """
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, z):
        """
        Realiza una pasada hacia atrás a través de todas las capas de la red.

        Parámetros:
        - z: Gradiente de la pérdida con respecto a la salida de la red.

        Retorna:
        - El gradiente de la pérdida con respecto a la entrada de la red.
        """
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z

    def update(self, lr):
        """
        Actualiza los parámetros de todas las capas de la red utilizando la tasa de aprendizaje especificada.

        Parámetros:
        - lr: Tasa de aprendizaje para actualizar los parámetros de las capas.
        """
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)