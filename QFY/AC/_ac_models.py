import numpy as np
from sklearn import linear_model, svm

from ..base import ProbCLFQuantifier, CrispCLFQuantifier
from ._base import ACModel


########################################################################################################################
# (P)AC Classes
########################################################################################################################

class AC(ACModel, CrispCLFQuantifier):

    def __init__(self, clf=svm.SVC(), n_folds=10):
        ACModel.__init__(self, clf)
        CrispCLFQuantifier.__init__(self, clf, n_folds)

    def _score_pos(self, X):
        return np.sum(self.clf.predict(X) == self.Y[1]) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pos_ind = y == self.Y[1]
        tpr = np.sum(y_scores[pos_ind]) / Y_cts[1]
        fpr = np.sum(y_scores[~pos_ind]) / Y_cts[0]
        return tpr, fpr


class PAC(ACModel, ProbCLFQuantifier):

    def __init__(self,
                 clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'),
                 n_folds=10):
        ACModel.__init__(self, clf)
        ProbCLFQuantifier.__init__(self, clf, n_folds)

    def _score_pos(self, X):
        return np.sum(self._clf_score(X)[:, 1]) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pos_ind = y == self.Y[1]
        tpr = np.sum(y_scores[pos_ind, 1]) / Y_cts[1]
        fpr = np.sum(y_scores[~pos_ind, 1]) / Y_cts[0]
        return tpr, fpr