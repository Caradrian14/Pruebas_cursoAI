import numpy as np

class Softmax:
    """
    Clase que implementa la función de activación Softmax.
    Softmax se utiliza comúnmente en la capa de salida de las redes neuronales para problemas de clasificación multiclase.
    Convierte un vector de valores en una distribución de probabilidad.
    """
    def forward(self, z):
        """
        Realiza una pasada hacia adelante aplicando la función Softmax.

        Parámetros:
        - z: Entrada a la capa Softmax. Debe ser un array de tamaño (batch_size, num_classes).

        Retorna:
        - Las probabilidades calculadas aplicando la función Softmax a la entrada.
        """
        self.z = z # Almacenar la entrada para usarla en la pasada hacia atrás

        # Calcular el máximo de cada fila y restarlo para mejorar la estabilidad numérica
        zmax = z.max(axis=1, keepdims=True)
        # Calcular la exponencial de los valores ajustados
        expz = np.exp(z - zmax)
        # Calcular la suma de las exponenciales para cada fila
        Z = expz.sum(axis=1, keepdims=True)
        # Calcular las probabilidades de Softmax
        return expz / Z

    def backward(self, dp):
        """
        Realiza una pasada hacia atrás a través de la capa Softmax para calcular los gradientes.

        Parámetros:
        - dp: Gradiente de la pérdida con respecto a la salida de la capa Softmax.

        Retorna:
        - El gradiente de la pérdida con respecto a la entrada de la capa Softmax.
        """
        # Calcular las probabilidades de Softmax para la entrada almacenada
        p = self.forward(self.z)
        # Calcular el producto elemento a elemento de las probabilidades y el gradiente de la pérdida
        pdp = p * dp
        # Calcular el gradiente de la pérdida con respecto a la entrada de la capa Softmax
        return pdp - p * pdp.sum(axis=1, keepdims=True)