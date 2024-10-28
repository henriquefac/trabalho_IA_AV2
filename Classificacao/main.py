import numpy as np
import matplotlib.pyplot as plt
from classes.dados.dados import DataHandler as dh

data = dh("EMGsDataset.csv")
data.setMatrizes()
print(data.y)
print(data.returnYasMatrix())
