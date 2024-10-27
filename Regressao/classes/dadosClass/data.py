import numpy as np

class Data():
    def __init__(self, path: str) -> None:
        self.path = path
        self.formatData()

    def formatData(self):
        data = np.loadtxt(self.path)
        x = data[:, 0]
        y = data[:, 1]
        x.shape = (len(x), 1)
        y.shape = (len(y), 1)
        
        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)
        self.x = x
        self.y = y


    def getData(self):
        return (self.x, self.y)

