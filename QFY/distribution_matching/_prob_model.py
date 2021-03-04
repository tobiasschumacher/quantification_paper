# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 01:49:22 2018

@author: tobi_
"""

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import cvxpy as cvx


class ED:

    def __init__(self):
        self.A = None
        self.B = None
        self.Y = None
        self.XY = None
        self.L = None

    def fit(self, X, y):

        self.Y = np.unique(y)
        self.L = len(self.Y)

        if self.L < 2:
            raise ValueError("There is only one unique value in target vector y.")

        Y_idx = [np.where(y == k) for k in self.Y]

        self.XY = [X[Y_idx[i]] for i in range(self.L)]

        self.A = np.zeros((self.L, self.L))

        for i in range(self.L):
            for j in range(i, self.L):

                Xi, Xj = self.XY[i], self.XY[j]
                ni, nj = Xi.shape[0], Xj.shape[0]
                self.A[i, j] = 1.0 / (ni * nj) * sum(np.sum(M) for M in pairwise_distances_chunked(Xi, Xj))
                if j > i:
                    self.A[j, i] = self.A[i, j]

        if self.L > 2:
            k = self.L - 1
            self.B = np.zeros((k, k))

            for i in range(k):
                for j in range(i, k):
                    self.B[i, j] = - self.A[i, j] + self.A[i, k] + self.A[k, j] - self.A[k, k]
                    if j > i:
                        self.B[j, i] = self.B[i, j]

        return self

    def predict(self, X):

        s = np.zeros(self.L)
        n = X.shape[0]

        for i in range(self.L):
            Xi = self.XY[i]
            ni = Xi.shape[0]
            s[i] = 1.0 / (ni * n) * sum(np.sum(M) for M in pairwise_distances_chunked(Xi, X))

        if self.L < 3:
            p = (s[1] - s[0] + self.A[0, 1] - self.A[1, 1]) / (-self.A[0, 0] + 2 * self.A[0, 1] - self.A[1, 1])

            if p < 0:
                return np.array([0, 1])
            if p > 1:
                return np.array([1, 0])

            return np.array([p, 1 - p])

        else:
            k = self.L - 1
            t = np.zeros(k)
            for i in range(k):
                t[i] = - s[i] + self.A[i, k] + s[k] - self.A[k, k]

            P = cvx.Variable(k)
            constraints = [P >= 0, cvx.sum(P) <= 1.0]
            problem = cvx.Problem(cvx.Minimize(cvx.quad_form(P, self.B) - 2 * P.T @ t), constraints)
            problem.solve()

            P = np.array(P.value).squeeze()
            return np.append(P, 1.0 - sum(P))
