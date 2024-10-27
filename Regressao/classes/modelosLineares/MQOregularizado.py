import numpy as np
from classes.modelosLineares.modeloBase import BaseModel as bm

class MQORegularizado(bm):
    def __init__(self, x: np.ndarray, y: np.ndarray, lambd: float = 0.0) -> None:
        super().__init__(x, y)
        self.lambd = lambd  # Parâmetro de regularização

    def trainModel(self):
        """
        Implementa a Regressão Linear Regularizada (MQO com Regularização L2).
        Calcula o vetor de coeficientes beta usando a fórmula de MQO com regularização.
        """
        # Cria uma matriz identidade do tamanho apropriado
        identity_matrix = np.eye(self.x.shape[1])  # Matriz identidade

        # Não penalize o primeiro coeficiente (intercepto)
        identity_matrix[0, 0] = 0  # Zera o valor correspondente ao intercepto

        # Calcula o vetor de coeficientes beta com regularização
        self.beta_hat = np.linalg.inv(self.x.T @ self.x + self.lambd * identity_matrix) @ self.x.T @ self.y
