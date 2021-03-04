# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:54:45 2018

@author: tobi_
"""

import numpy as np
from itertools import combinations_with_replacement


def count_target_prevalences(y, Y=None):
    if Y is None:
        return np.unique(y, return_counts=True)[1]

    else:
        YT = np.unique(y, return_counts=True)
        L = len(Y)
        LT = len(YT[0])
        if L == LT and all(np.equal(Y, YT[0])):
            return YT[1]
        else:
            i = 0
            y_ct = np.zeros(L)
            for j in range(L):

                if YT[0][i] == Y[j]:
                    y_ct[j] = YT[1][i]
                    i += 1
                    if i >= LT:
                        break

            if i < LT:
                raise ValueError("Elements of vector y do not match target space Y.")

            return y_ct


def rel_target_prevalences(y, Y=None):
    m = np.size(y)
    return count_target_prevalences(y, Y) / m


def partitions(n, b):
    masks = np.identity(b, dtype=int)
    for c in combinations_with_replacement(masks, n):
        yield sum(c)


def distributions(n_classes, den):
    return 1.0 / den * np.array(list(partitions(den, n_classes)))
