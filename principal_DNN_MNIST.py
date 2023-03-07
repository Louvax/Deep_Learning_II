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
            X_copy, y_copy = X.copy(), y.copy()
            assert len(X_copy) == len(y_copy)
            p = np.random.permutation(len(X_copy))
            X_copy, y_copy = X_copy[p], y_copy[p]
            for batch in range(0, len(X_copy), batch_size):
                X_batch = X_copy[batch : min(batch + batch_size, len(X_copy))]
                y_batch = y_copy[batch : min(batch + batch_size, len(y_copy))]
                tb = len(X_batch)
                z = self.entree_sortie_reseau(X_batch)

                dz1 = z[-1] - y_batch
                grad_b = (1 / tb) * np.sum(dz1)
                grad_w = (1 / tb) * (z[-2].T @ dz1)

                self.RBM.W -= lr * grad_w
                self.RBM.b -= lr * grad_b

                dz2 = z[-2] * (1 - z[-2])
                dz1 = (dz2 @ self.RBM.W.T) * dz2

                grad_w = z[-3].T @ dz1 / tb
                grad_b = np.sum(dz1) / tb

                self.DBN.Dbn[-1].W -= lr * grad_w
                self.DBN.Dbn[-1].b -= lr * grad_b

                for layer in range(len(self.DBN) - 3, -1, -1):
                    if layer == 0:
                        data = X_batch
                    else:
                        data = z[layer - 1]

                    dz2 = z[layer] * (1 - z[layer])
                    dz1 = (dz2 @ self.DBN.Dbn[layer + 1].W.T) * dz2

                    grad_w = z[layer - 1].T @ dz1 / tb
                    grad_b = np.sum(dz1) / tb

                    self.DBN.Dbn[layer].W -= lr * grad_w
                    self.DBN.Dbn[layer].b -= lr * grad_b

                z = self.entree_sortie_reseau(X_batch)

            crossentropy = -np.mean(np.log10(z)[y_batch == 1])

            print(f"Epoch : {epoch}, Loss : {crossentropy}")

    def test_DNN(self, X, y):
        y_pred = self.entree_sortie_reseau(X)[-1]

        crossentropy = -np.mean(np.log10(y_pred)[y == 1])

        for y_i in y_pred:
            predicted = np.zeros(len(y_i))
            predicted[np.argmax(y_i)] = 1

            y_i = predicted

        accuracy = (y_pred == y) / len(y) * 100

        print(f"Accuracy : {accuracy}, CrossEntropy : {crossentropy}")

        return accuracy / 100
