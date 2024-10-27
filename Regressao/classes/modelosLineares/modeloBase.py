import numpy as np

class BaseModel():
    """
    Modelo base para as classes que implementam os
    modelos de regressão linear. Ele recebe no seu construtor
    a matriz X de variáveis independentes com o intercepto e 
    a matriz y de variáveis dependentes de x, respectivo a posição.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.beta_hat:np.ndarray
    # função para ser implementada nos modelos
    def trainModel(self):
        pass

    def predict(self, x_new: np.ndarray):
        """
        Faz previsões com base em novos dados x_new.
        
        :param x_new: matriz de dados (n X p) sem a coluna de intercepto
        :return: previsões baseadas em beta_hat
        """
        # Verifica se o modelo foi treinado
        if self.beta_hat is None:
            raise ValueError("O modelo não foi treinado. Chame trainModel() antes de predict().")
        return x_new @ self.beta_hat