from classes.dadosClass.data import Data
import numpy as np

class MonteCarlo(Data):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    # Dados para particionar 80 | 20
    def partitionData(self):
        """
        Função que retorna a massa de dados dividida entre dados de treinamento e dados de teste
        na proporção 80/20, de forma aleatória.
        O retorno é uma tupla de matriz no formato n X p+2 onde a última coluna são os dados dependentes
        e a primeira uma coluna de uns para o intercepto.
        A primeira matriz é referente aos dados de treinamento e a segunda aos dados de teste.
        """
        # Criar matriz de dados x e y
        x, y = self.getData()
        matriz_dados = np.concatenate((x, y), axis=1)

        # Número de amostras
        n = matriz_dados.shape[0]
        
        # Lista de índices permutados
        index = np.random.permutation(np.arange(n))  # Correção aqui
        
        # Ponto de divisão 80 | 20
        num_div = int((n * 80) / 100)
        
        # Dados de treinamento e teste
        train_data = matriz_dados[index[:num_div], :]
        test_data = matriz_dados[index[num_div:], :]

        return train_data, test_data

    # Separa uma matriz de dados entre dados independentes e dados dependentes
    @staticmethod
    def formatMatrix(matriz_dados: np.ndarray):
        """
        Recebe uma matriz de dados e retorna duas matrizes:
        - uma com as variáveis independentes (todas as colunas exceto a última),
        - e outra com a variável dependente (última coluna).
        """
        X = matriz_dados[:, :-1]  # Todas as colunas exceto a última
        y = matriz_dados[:, -1]   # Apenas a última coluna
        return X, y.reshape(-1, 1)  # Formata y como coluna

