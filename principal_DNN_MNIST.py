import numpy as np
from principal_RBM_alpha import *
from principal_DBN_alpha import *


class DNN:
    def __init__(self, config):
        self.init_DNN(config)

    def init_DNN(self, config):
        self.DBN = DBN(config[:-1])
        self.RBM = RBM(config[-2], config[-1])

    def pretrain_DNN(self, X, epochs, batch_size, lr):
        self.DBN.train_DBN(X, epochs, batch_size, lr)

    def calcul_softmax(self, X):
        x = self.RBM.entree_sortie_RBM(X)
        return np.exp(x) / sum(np.exp(x))

    def entree_sortie_reseau(self, X):
        output = []
        X_copy = X.copy()
        for rbm in self.DBN.Dbn:
            output.append(rbm.entree_sortie_RBM(X_copy))
            X_copy = rbm.entree_sortie_RBM(X_copy)

        output.append(self.calcul_softmax(X_copy))

        return output
    
    def retropropagation(self, X, y, epochs, batch_size, lr):
        for epoch in range(epochs):
            X_copy = X.copy()
            np.random.shuffle(X_copy)
            for batch in range(0, len(X_copy), batch_size):
                X_batch = X_copy[batch:min(batch+batch_size, len(X_copy))]
                tb = len(X_batch)

                

