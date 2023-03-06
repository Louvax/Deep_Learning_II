import scipy.io
import numpy as np


def lire_alpha_digit(idx):
    mat = scipy.io.loadmat("./data/binaryalphadigs.mat")
    data = mat["dat"][idx]
    output = np.reshape(data[0], 20 * 16)
    for im in data[1:]:
        output = np.vstack((output, np.reshape(im, 20 * 16)))

    return output


class RBM:
    def __init__(self, p, q):
        self.init_RBM(p, q)

    def init_RBM(self, p, q):
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=0.1, size=(p, q))

    def entree_sortie_RBM(self, X):
        return 1 / (1 + np.exp(-(X @ self.W + self.b)))

    def sortie_entree_RBM(self, H):
        return 1 / (1 + np.exp(-(H @ self.W.T + self.a)))

    def train_RBM(self, X, epochs, batch_size, lr):
        q, p = len(self.b), len(self.a)
        for epoch in range(epochs):
            X_copy = X.copy()
            np.random.shuffle(X_copy)
            for batch in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[batch : min(batch + batch_size, X_copy.shape[0])]
                tb = len(X_batch)
                v0 = X_batch
                Ph_v0 = self.entree_sortie_RBM(v0)
                h0 = (np.random.rand(tb, q) < Ph_v0) * 1
                Pv_h0 = self.sortie_entree_RBM(h0)
                v1 = (np.random.rand(tb, p) < Pv_h0) * 1
                Ph_v1 = self.entree_sortie_RBM(v1)

                grad_a = np.sum(v0 - v1, axis=0)
                grad_b = np.sum(Ph_v0 - Ph_v1, axis=0)
                grad_w = v0.T @ Ph_v0 - v1.T @ Ph_v1

                self.W += lr / tb * grad_w
                self.a += lr / tb * grad_a
                self.b += lr / tb * grad_b
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            print(np.sum(X - X_rec) ** 2 / X_copy.shape[0])

    def generate_data(self, nb_data, nb_gibbs):
        data = []
        for i in range(nb_data):
            v = (np.random.rand(len(self.a)) < 1 / 2) * 1
            for iter in range(nb_gibbs):
                h = (np.random.rand(len(self.b)) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(len(self.a)) < self.sortie_entree_RBM(h)) * 1
            data.append(v)
        return np.array(data)
