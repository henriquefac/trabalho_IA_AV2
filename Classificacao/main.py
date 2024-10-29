import numpy as np
import matplotlib.pyplot as plt
from classes.dados.dados import DataHandler as dh
from classes.dados.monteCarlo import MonteCarlo as mc
from classes.modelos.MQOtradicional import MQOT 

data = dh(r"notebooks\EMGsDataset.csv")
data.setMatrizes()
data.returnYasMatrix()
x, y = data.x, data.matY
print(x.T)
print(y.T)
x_train, y_train, x_test, y_test = mc.partition(x, y)

modelo = MQOT(x_train, y_train)
modelo.trainModel()
y_predict = modelo.predict(x_test)



y_test_classes = np.argmax(y_test, axis=1) + 1
dif_y = y_test_classes - y_predict
taxa_acerto = np.count_nonzero(dif_y == 0) / len(y_test_classes.flatten())


print(taxa_acerto)

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Matriz de confusão
cm = confusion_matrix(y_test_classes, y_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test_classes), yticklabels=np.unique(y_test_classes))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()