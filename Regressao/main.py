import numpy as np
import matplotlib.pyplot as plt
from classes.dadosClass.monteCarlo import MonteCarlo as mc
from classes.modelosLineares.MQOtradicional import MQOTradicional as MQOT

# Pegar dados de treino e dados de teste
dados = mc(r"notebooks\aerogerador.dat")
treino, teste = dados.partitionData()

# Formatar os dados
x_train, y_train = mc.formatMatrix(treino)
x_test, y_test = mc.formatMatrix(teste)

# Verificar se os dados foram carregados corretamente
if x_train.shape[0] == 0 or x_test.shape[0] == 0:
    raise ValueError("Os dados de treino ou teste estão vazios.")

# Plotar dados de teste original
plt.scatter(x_test[:, 1:], y_test, color='purple', label='Dados de Teste')

# Realizar o modelo
model = MQOT(x_train, y_train)
model.trainModel()

# Fazer predições
new_y = model.predict(x_test)

# soma dos desvios quadrados

soma_desvios_quadrados = np.sum((new_y.flatten() - y_test.flatten()) ** 2)
print(soma_desvios_quadrados)
# Plotar as predições
plt.scatter(x_test[:, 1:], new_y, color='blue', alpha=0.5, label='Predições do Modelo')

# Configurar o gráfico
plt.title('Predições do Modelo MQO')
plt.xlabel('Variáveis Independentes')
plt.ylabel('Variáveis Dependentes')
plt.legend()
plt.show()
