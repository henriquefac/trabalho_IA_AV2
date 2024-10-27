import numpy as np
from classes.modelosLineares.modeloBase import BaseModel as bm
class MQOTradicional(bm):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__(x, y)

    def trainModel(self):
        """
        Implementa a Regressão Linear por Mínimos Quadrados Ordinários (MQO).
        Calcula o vetor de coeficientes beta usando a fórmula de MQO.
        """

        self.beta_hat = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

    