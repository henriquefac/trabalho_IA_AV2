import numpy as np

from classes.modelos.modeloBase import BaseModelClass

# sempre lmebrar, para gausiana
# X E R{p x n}


class GausNaiveBay(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        super().__init__(x, y)
        # lista de classes
        self.c = c
        self.separatedDataBase()

    # é preciso calcular a ,édia e desvio padrão para cada classe
    def getDataByClass(self, classe: int):
        # Selecionar as colunas de `y` onde a classe corresponde
        indices_classe = np.where(self.y[0, :] == classe)[0]  # Obter os índices para a classe específica
        # Selecionar as amostras em `x` que correspondem à classe desejada
        x_classe = self.x[:, indices_classe]
        y_classe = self.y[:, indices_classe]
        return x_classe, y_classe
    
    def separatedDataBase(self):
        self.separeted_matrix = {}
        for classe in self.c:
            x_c, y_c = self.getDataByClass(classe)
            self.separeted_matrix[classe] = {
                "x": x_c,
                "y": y_c
            }


    def getStatistcs(self):
        self.statics = {}
        for classe in self.c:
            mean = np.mean(self.separeted_matrix[classe]["x"], axis=1).reshape(-1, 1)
            epsilon = 1e-10
            var = np.var(self.separeted_matrix[classe]["x"], axis=1).reshape(-1, 1) + epsilon
            prior = self.separeted_matrix[classe]["x"].shape[1] / self.x.shape[1]
            self.statics[classe] = {
                "mean": mean,
                "var": var,
                "prior": prior
            }



    def discriminante(self, x_new, classe):
        stat = self.statics[classe]
        mean = stat["mean"]
        var = stat["var"]
        prior = stat["prior"]

        # Cálculo da probabilidade de ser da classe dada a nova amostra
        # Usando a fórmula da função de densidade da distribuição normal
        n = x_new.shape[0]
        probabilidade = prior  # Começa com a probabilidade a priori

        for i in range(n):
            try:
                # Aplicando a função de densidade de probabilidade da normal
                termo1 = 1 / np.sqrt(2 * np.pi * var[i])
                termo2 = np.exp(-((x_new[i] - mean[i]) ** 2) / (2 * var[i]))
                probabilidade *= termo1 * termo2
            except:
                print(classe)
                # Aplicando a função de densidade de probabilidade da normal
                termo1 = 1 / np.sqrt(2 * np.pi * var[i])
                termo2 = np.exp(-((x_new[i] - mean[i]) ** 2) / (2 * var[i]))
                probabilidade *= termo1 * termo2
        return probabilidade

    

    def predict(self, x_new: np.ndarray):
        predictions = []

        for individuo in x_new.T:

            max_score = - np.inf
            predicted_classs = None
            for classe in self.c:
                score = self.discriminante(individuo.reshape(-1, 1), classe)
                if score > max_score: 
                    max_score = score
                    predicted_classs = classe
            
            predictions.append(predicted_classs)
        return predictions