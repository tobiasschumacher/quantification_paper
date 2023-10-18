import numpy as np
from sklearn import preprocessing, svm, linear_model
from copy import deepcopy
from ._base import BinaryAC, BinaryPAC, BinaryTSX, BinaryTSMax, BinaryTS50, BinaryMS
from ..base import Quantifier
import warnings


class ACQuantifier(Quantifier):

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
            curr_qf.fit(X, y, Y_cts)
            self.qf_models.append(curr_qf)
        else:
            lb = preprocessing.LabelBinarizer()
            y = lb.fit_transform(y)
            for i in range(self.L):
                curr_cts = [np.array([0, 1]), np.array([n - Y_cts[1][i], Y_cts[1][i]])]
                curr_qf = deepcopy(self.qf)
                curr_qf.fit(X, y[:, i], curr_cts)
                self.qf_models.append(curr_qf)

        return self

    def predict(self, X):

        if self.L == 2:
            p = self.qf_models[0].predict(X)
            return np.array([1.0 - p, p])
        else:
            p = np.zeros(self.L)
            for i in range(self.L):
                p[i] = self.qf_models[i].predict(X)

            p_sum = np.sum(p)

            if p_sum == 0:
                warnings.warn("OVR Quantifier estimated prevalence of every class as 0. "
                              "Therefore, uniform distribution was returned.")
                return np.array([1.0/self.L]*self.L)

            return p / p_sum


class AC(ACQuantifier):
    def __init__(self, clf=svm.SVC(), n_folds=10):
        ACQuantifier.__init__(self, qf=BinaryAC(clf=clf, n_folds=n_folds))


class PAC(ACQuantifier):
    def __init__(self,
                 clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'),
                 n_folds=10):
        ACQuantifier.__init__(self, qf=BinaryPAC(clf=clf, n_folds=n_folds))


class TSX(ACQuantifier):
    def __init__(self,
                 clf=svm.SVC(),
                 n_folds=10,
                 precision=3,
                 predict_proba=None):
        ACQuantifier.__init__(self, qf=BinaryTSX(clf=clf, n_folds=n_folds, precision=precision,
                                                 predict_proba=predict_proba))


class TS50(ACQuantifier):
    def __init__(self,
                 clf=svm.SVC(),
                 n_folds=10,
                 precision=3,
                 predict_proba=None):
        ACQuantifier.__init__(self, qf=BinaryTS50(clf=clf, n_folds=n_folds, precision=precision,
                                                  predict_proba=predict_proba))


class TSMax(ACQuantifier):
    def __init__(self,
                 clf=svm.SVC(),
                 n_folds=10,
                 precision=3,
                 predict_proba=None):
        ACQuantifier.__init__(self, qf=BinaryTSMax(clf=clf, n_folds=n_folds, precision=precision,
                                                   predict_proba=predict_proba))


class MS(ACQuantifier):
    def __init__(self,
                 clf=svm.SVC(),
                 n_folds=10,
                 precision=3,
                 delta_min=0.25,
                 predict_proba=None):
        ACQuantifier.__init__(self, qf=BinaryMS(clf=clf, n_folds=n_folds, precision=precision, delta_min=delta_min,
                                                predict_proba=predict_proba))
