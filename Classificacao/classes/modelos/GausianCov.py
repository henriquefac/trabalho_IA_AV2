import numpy as np

from classes.modelos.modeloBase import BaseModelClass

# para cada classe, é preciso criar o vetor de médias das features
# e a matriz de covariancia
class GaussianCovModel(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        super().__init__(x, y)
        # lista de classes
        self.c = c
        self.separatedDataBase()
        
        # Separar matriz com base nas classes
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

    # criar statistias dos dados
    # como a matrix de covariancia é igaul para todos os grupos de dados
    # ela é feita somente uma vez
    def getStatistcs(self):
        self.cov = np.cov(self.x)
        self.inv_cov = np.linalg.inv(self.cov)
        self.mean_by_class = {}
        for i in self.c:
            mean = np.mean(self.separeted_matrix[i]["x"], axis=1).reshape(-1,1)
            self.mean_by_class[i] = mean


    def descriminante(self, x_new, classe):

        # Calculando o termo 2
        diff = x_new - self.mean_by_class[classe]
        termo_2 =  (diff.T @ self.inv_cov @ diff)

        return termo_2

    def predict(self, x_new: np.ndarray):
        predictions = []

        for individuo in x_new.T:
            # buscamos o menor para consweguir 
            # minimizar a distância de Mahalanobis (LDA)
            min_score = np.inf
            predicted_classs = None
            for classe in self.c:
                score = self.descriminante(individuo.reshape(-1, 1), classe)

                if score < min_score: 
                    min_score = score
                    predicted_classs = classe
            
            predictions.append(predicted_classs)

        return predictions
        