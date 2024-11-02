import numpy as np

from classes.modelos.modeloBase import BaseModelClass

# sempre lmebrar, para gausiana
# X E R{p x n}


class GausianFriedman(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, lamb) -> None:
        super().__init__(x, y)
        # lista de classes
        self.c = c
        self.lamb = lamb
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
                "invCov": None,
                "detCov":None,
                "prior":prior
            }
        self.setCov()

    def getCovAgregado(self):
        var = 0
        for classe in self.c:
            # matrix de covariância da classe
            cov_c = self.statics[classe]["cov"]
            # numero de amostras da clase
            n_c = self.separeted_matrix[classe]["x"].shape[1]
            # matriz de covariância vezes número de amostras da classe 
            # acumulada para cada classe
            var += cov_c * n_c
        # média ponderada das matrizes de covariância
        return var / self.x.shape[1]

    def setCov(self):
        # gerar matriz de covariância agregada
        gre_cov = self.getCovAgregado()
        for classe in self.c:
            # número de amostras
            N = self.x.shape[1]
            # número de amostras da classe
            n = self.separeted_matrix[classe]["x"].shape[1]
            # matriz de covariância (tradicional)
            cov = self.statics[classe]["cov"]
            # formula para calcular nova matrix de covariância regularizada
            # por lambda
            new_cov = ((1- self.lamb)*(n * cov) + (self.lamb * N * gre_cov))/((1-self.lamb)*n + self.lamb*N)
            self.statics[classe]["cov"] = new_cov
            self.statics[classe]["invCov"] = np.linalg.inv(new_cov)
            self.statics[classe]["detCov"] = np.linalg.det(new_cov)

    def descriminante(self, x_new, classe):
        termo_1 = -0.5 * np.log(self.statics[classe]["detCov"])

        # Calculando o termo 2
        diff = x_new - self.statics[classe]["mean"]
        termo_2 = -0.5 * (diff.T @ self.statics[classe]["invCov"] @ diff)

        return termo_1 + termo_2




    def predict(self, x_new: np.ndarray):
        predictions = []

        desciminante = self.descriminante

        for individuo in x_new.T:

            max_score = - np.inf
            predicted_classs = None
            for classe in self.c:
                score = desciminante(individuo.reshape(-1, 1), classe)

                if score > max_score: 
                    max_score = score
                    predicted_classs = classe
            
            predictions.append(predicted_classs)
        return predictions