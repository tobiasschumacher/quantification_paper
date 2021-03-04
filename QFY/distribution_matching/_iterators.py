# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:31:21 2018

@author: tobi_
"""

from ._base import *


class MMIterator(ProbCLFQuantifier, ABC):

    def __init__(self,
                 clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'),
                 eps=1e-06,
                 max_iter=1000):

        ProbCLFQuantifier.__init__(self, clf=clf, nfolds=0)
        self.Y = None
        self.Y_rates = None
        self.eps = eps
        self.max_iter = max_iter

    def _fit(self, X, y, Y_cts):

        self.Y = Y_cts[0]
        self.Y_rates = Y_cts[1] * 1.0 / len(y)

        # now fit real classifier
        self.clf.fit(X, y)

    def fit(self, X, y):
        Y_cts = list(np.unique(y, return_counts=True))
        self._fit(X, y, Y_cts)
        return self


class EM(MMIterator):

    def predict(self, X):

        m = X.shape[0]
        yp = self._clf_score(X)

        p_new = self.Y_rates
        p_old = np.ones(self.Y_rates.shape)

        n_it = 0

        while (np.linalg.norm(p_old - p_new) > self.eps) and n_it < self.max_iter:
            p_old = np.array(p_new)
            CM = np.array([p_old / self.Y_rates * yp[i] for i in range(m)])
            CM = CM / (np.array([np.sum(CM, axis=1)]).transpose())

            p_new = 1.0 / m * np.sum(CM, axis=0)
            n_it += 1

        return p_new


class BinaryCDE(MMIterator):

    def predict(self, X):
        yp = self._clf_score(X)
        c = np.ones(2)
        c_old = np.zeros(2)

        q = 2
        n_it = 0

        while np.linalg.norm(c - c_old) > self.eps and n_it <= self.max_iter:
            y = np.apply_along_axis(lambda p: self.Y[1] if p[1] > c[0] / np.sum(c) else self.Y[0], axis=1, arr=yp)
            c_old = np.copy(c)
            q = rel_target_prevalences(y, self.Y)[1]

            c[0] = (1 - q) / (self.Y_rates[0])
            c[1] = q / self.Y_rates[1]
            n_it += 1

        if n_it >= self.max_iter:
            warnings.warn("The CDE iteration has not converged")

        return np.array([1 - q, q])
