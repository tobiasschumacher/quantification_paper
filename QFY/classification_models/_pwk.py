# -*- coding: utf-8 -*-

from ._base import *
from sklearn.neighbors import NearestNeighbors


class PWKCLF:

    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):

        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                     algorithm=algorithm,
                                     leaf_size=leaf_size,
                                     metric=metric,
                                     p=p,
                                     metric_params=metric_params,
                                     n_jobs=n_jobs)

        self.Y = None
        self.Y_map = None
        self.w = None
        self.y = None
        self.n_neighbors = n_neighbors

        if alpha < 1:
            raise ValueError("alpha must not be smaller than 1")
        else:
            self.alpha = alpha

    def fit(self, X, y):

        n = X.shape[0]
        if n < self.n_neighbors:
            self.nbrs.set_params(n_neighbors=n)

        self.y = y
        Y_cts = np.unique(y, return_counts=True)
        self.Y = Y_cts[0]
        self.Y_map = dict(zip(self.Y, range(len(self.Y))))

        self.w = (Y_cts[1] / np.min(Y_cts[1])) ** (-1.0/self.alpha)
        self.nbrs.fit(X)

        return self

    def predict(self, X):

        N = X.shape[0]
        nn_ind = self.nbrs.kneighbors(X, return_distance=False)

        CM = np.zeros(shape=(N, len(self.Y)))

        for i in range(N):
            for j in nn_ind[i]:
                CM[i, self.Y_map[self.y[j]]] += 1

        CM = np.multiply(CM, self.w)

        return np.array([self.Y[i] for i in np.apply_along_axis(np.argmax, axis=1, arr=CM)])


class PWK(CC):

    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):

        CC.__init__(self, clf=PWKCLF(alpha=alpha,
                                     n_neighbors=n_neighbors,
                                     algorithm=algorithm,
                                     metric=metric,
                                     leaf_size=leaf_size,
                                     p=p,
                                     metric_params=metric_params,
                                     n_jobs=n_jobs))
