from ._base import *
from sklearn import preprocessing
from copy import deepcopy
from ._clf_models import BinaryDyS
from ._iterators import BinaryCDE


class OVRQuantifier(Quantifier):

    def __init__(self, qf):
        Quantifier.__init__(self)
        self.qf = qf
        self.qf_models = []
        self.L = None

    def fit(self, X, y):

        Y_cts = np.unique(y, return_counts=True)
        self.Y = Y_cts[0]
        n = len(y)
        self.L = len(self.Y)

        if self.L == 2:
            curr_qf = deepcopy(self.qf)
            curr_qf._fit(X, y, Y_cts)
            self.qf_models.append(curr_qf)
        else:
            lb = preprocessing.LabelBinarizer()
            y = lb.fit_transform(y)
            for i in range(self.L):
                curr_cts = [np.array([0, 1]), np.array([n - Y_cts[1][i], Y_cts[1][i]])]
                curr_qf = deepcopy(self.qf)
                curr_qf._fit(X, y[:, i], curr_cts)
                self.qf_models.append(curr_qf)

        return self

    def predict(self, X):

        if self.L == 2:
            return self.qf_models[0].predict(X)
        else:
            p = np.zeros(self.L)
            for i in range(self.L):
                p[i] = self.qf_models[i].predict(X)[1]

            p_sum = np.sum(p)
            if p_sum == 0:
                warnings.warn("OVR Quantifier estimated prevalence of every class as 0. "
                              "Therefore, uniform distribution was returned.")
                return np.array([1.0 / self.L] * self.L)

            return p/p_sum


class DyS(OVRQuantifier):

    def __init__(self, clf=svm.SVC(), distance="TS", nbins=10, nfolds=10, solve_cvx=True):
        OVRQuantifier.__init__(self, qf=BinaryDyS(clf=clf,
                                                  distance=distance,
                                                  nbins=nbins,
                                                  nfolds=nfolds,
                                                  solve_cvx=solve_cvx
                                                  ))


# ----------------------------------------------------------------------------------------------------------------------
# Forman's Mixture Model (ultimately a L1-Minimizer with many bins)
# ----------------------------------------------------------------------------------------------------------------------
class FormanMM(DyS):

    def __init__(self, clf=svm.LinearSVC(), nbins=100, nfolds=10):
        DyS.__init__(self, clf=clf, distance="L1", nfolds=nfolds, nbins=nbins)


class CDE(OVRQuantifier):

    def __init__(self,
                 clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'),
                 eps=1e-06,
                 max_iter=1000):

        OVRQuantifier.__init__(self, qf=BinaryCDE(clf=clf, eps=eps, max_iter=max_iter))
