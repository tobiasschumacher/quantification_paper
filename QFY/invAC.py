from abc import ABC, abstractmethod
import numpy as np
from QFY.base import CLFQuantifier, CrispCLFQuantifier, ScoreCLFQuantifier, ProbCLFQuantifier
from sklearn import linear_model

class InvACModel(CLFQuantifier, ABC):

    def __init__(self):
        self.tr, self.fr = None, None

    @abstractmethod
    def _get_rates(self, y, y_scores, Y_cts):
        pass

    @abstractmethod
    def _score_pos(self, X):
        pass

    def fit(self, X, y):

        Y_cts = np.unique(y, return_counts=True)
        self.Y = Y_cts[0]
        Y_cts = Y_cts[1]

        nfolds = min(self.nfolds, min(Y_cts))
        y_scores = self._cv_score(X, y, nfolds)

        self.tr, self.fr = self._get_rates(y, y_scores, Y_cts)

    def predict(self, X):

        y_pos = self._score_pos(X)

        p = min(1.0, max(0.0,self.tr*y_pos + self.fr*(1-y_pos)))

        return np.array([1-p, p])


class InvAC(InvACModel, CrispCLFQuantifier):

    def __init__(self, clf, nfolds):
        InvACModel.__init__(self)
        CrispCLFQuantifier.__init__(self, clf, nfolds)

    def _score_pos(self, X):
        return np.sum(self.clf.predict(X) == self.Y[1]) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pos_ind = y == self.Y[1]
        tpr = np.sum(y_scores[pos_ind]) / Y_cts[1]
        fpr = np.sum(y_scores[~pos_ind]) / Y_cts[0]
        return tpr, fpr


class InvPAC(InvACModel, ProbCLFQuantifier):

    def __init__(self, clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'), nfolds=10):
        InvACModel.__init__(self)
        ProbCLFQuantifier.__init__(self, clf, nfolds)

    def _score_pos(self, X):
        return np.sum(self._clf_score(X)[:, 1]) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pos_ind = y == self.Y[1]
        tpr = np.sum(y_scores[pos_ind, 1]) / np.sum(y_scores[:, 1])
        fpr = np.sum(y_scores[pos_ind, 0]) / np.sum(y_scores[:, 0])
        return tpr, fpr
