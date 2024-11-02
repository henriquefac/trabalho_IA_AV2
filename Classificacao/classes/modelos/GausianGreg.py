import numpy as np

from classes.modelos.modeloBase import BaseModelClass

# sempre lmebrar, para gausiana
# X E R{p x n}


class GausianGredModel(BaseModelClass):
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


    # para cada classe, calcular 
    def getStatistcs(self):
        self.statics = {}
        for classe in self.c:
            # media
            mean = np.mean(self.separeted_matrix[classe]["x"], axis=1).reshape(-1, 1)
            
            cov = np.cov(self.separeted_matrix[classe]["x"])
            # prior_proba
            prior = (self.separeted_matrix[classe]["x"].shape[1])/self.x.shape[1]
            self.statics[classe] = {
                "mean":mean,
                "cov":cov,
                "prior":prior
            }



    def descriminante(self, x_new, classe):

        # Calculando o termo 2
        diff = x_new - self.statics[classe]["mean"]
        termo_2 = (diff.T @ self.inv_cov @ diff)

        return termo_2

    
    def getCovAgregado(self):
        var = 0
        for classe in self.c:
            cov_c = self.statics[classe]["cov"]
            n_c = self.separeted_matrix[classe]["x"].shape[1]
            var += cov_c * n_c
        return var / self.x.shape[1]

    def predict(self, x_new: np.ndarray):
        predictions = []

        self.cov = self.getCovAgregado()
        self.inv_cov = np.linalg.inv(self.cov)

        for individuo in x_new.T:
            # buscamos o menor valor vizando
            # minimizar as distâncias de Mahalanobis (LDA)
            min_score = np.inf
            predicted_classs = None
            for classe in self.c:
                score = self.descriminante(individuo.reshape(-1, 1), classe)

                if score < min_score: 
                    min_score = score
                    predicted_classs = classe
            
            predictions.append(predicted_classs)
        return predictions