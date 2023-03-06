import scipy.io
import numpy as np
from principal_RBM_alpha import *


class DBN:
    def __init__(self, config):
        self.init_DBN(config)

    def init_DBN(self, config):

        self.Dbn = []

        for i in range(len(config) - 1):
            self.Dbn.append(RBM(config[i], config[i + 1]))

    def train_DBN(self, X, batch_size, epochs, lr):
        for epoch in range(epochs):
            X_copy = X.copy()
            for i in range(len(self.Dbn)):
                self.Dbn[i].train_RBM(X_copy, 1, batch_size, lr)
                X_copy = self.Dbn[i].entree_sortie_RBM(X_copy)

    def generate_image_DBN(self, nb_data, nb_gibbs):
        v = self.Dbn[-1].generate_data(nb_data, nb_gibbs)
        for i in range(len(self.Dbn) - 2, -1, -1):
            v = (
                np.random.rand(nb_data, len(self.Dbn[i].a))
                < self.Dbn[i].sortie_entree_RBM(v)
            ) * 1
        return v
