import numpy as np

"""
Esta función matemática de pérdida es comúnmente utilizada en problemas de clasificación.
"""
class CrossEntropyLoss:

    def forward(self,p,y):
        """
        Calcula la pérdida de entropía cruzada hacia adelante.

        Parámetros:
        - p: Array de probabilidades predichas por el modelo. Cada fila representa una distribución de probabilidad sobre las clases.
        - y: Array de etiquetas verdaderas. Cada valor es el índice de la clase verdadera para la muestra correspondiente.

        Retorna:
        - El valor de la pérdida de entropía cruzada promedio sobre todas las muestras de entrada.
        """
        self.probabilidad = p  # Almacenar las probabilidades predichas
        self.y = y  # Almacenar las etiquetas verdaderas

        # Seleccionar las probabilidades correspondientes a las clases verdaderas
        probabilidad_of_y = p[np.arange(len(y)), y]

        # Calcular el logaritmo de las probabilidades seleccionadas
        logaritmo_probabilidad = np.log(probabilidad_of_y)

        # Retornar la pérdida promedio (negativa del logaritmo de la probabilidad)
        return -logaritmo_probabilidad.mean()

    def backward(self, loss):
        """
        Calcula el gradiente de la pérdida de entropía cruzada con respecto a las probabilidades predichas.

        Parámetros:
        - loss: Valor de la pérdida calculada en el paso hacia adelante.

        Retorna:
        - El gradiente de la pérdida con respecto a las probabilidades predichas.
        """
        # Inicializar el gradiente con ceros, con la misma forma que las probabilidades predichas
        dlog_softmax = np.zeros_like(self.probabilidad)

        # Calcular el gradiente para las clases verdaderas
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0 / len(self.y)

        # Dividir el gradiente por las probabilidades predichas para obtener el gradiente final
        return dlog_softmax / self.probabilidad