from abc import ABC, abstractmethod
import numpy as np
from QFY.base import CLFQuantifier, ScoreCLFQuantifier


########################################################################################################################
# AC Base Class
########################################################################################################################

class ACModel(CLFQuantifier, ABC):

    def __init__(self, clf, tpr=None, fpr=None):
        CLFQuantifier.__init__(self, clf)
        self.tpr, self.fpr = tpr, fpr
        self.n_folds = None

    @abstractmethod
    def _get_rates(self, y, y_scores, Y_cts):
        pass

    @abstractmethod
    def _score_pos(self, X):
        pass

    def fit(self, X, y):
        self.Y, Y_cts = np.unique(y, return_counts=True)
        if len(self.Y) > 2:
            return ValueError(
                "Adjusted count-based methods only works for binary quantification. Multiclass "
                "quantification is possible via OVRQuantifier class, but not recommended due to "
                "theoretical issues with that approach."
            )

        n_folds = min(self.n_folds, min(Y_cts))
        y_scores = self._cv_score(X, y, n_folds)

        self.tpr, self.fpr = self._get_rates(y, y_scores, Y_cts)

        return self

    def predict(self, X):
        y_pos = self._score_pos(X)
        delta = self.tpr - self.fpr

        if delta == 0:
            return np.array([1.0-y_pos, y_pos])

        p = (y_pos - self.fpr) / delta

        p = np.clip(p, a_min=0., a_max=1.)

        return np.array([1.0-p, p])


########################################################################################################################
# ThresholdModel/Selector Base Classes
########################################################################################################################

class ThresholdModel(ACModel, ScoreCLFQuantifier):

    def __init__(self, clf, n_folds, tpr=None, fpr=None, threshold=None, predict_proba=None):
        ACModel.__init__(self, clf, tpr, fpr)
        ScoreCLFQuantifier.__init__(self, clf=clf, n_folds=n_folds, predict_proba=predict_proba)

        self.threshold = threshold

    def _score_pos(self, X):
        return np.sum(self._clf_score(X) >= self.threshold) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pass


# Threshold Selector Base Class
class ThresholdSelector(ThresholdModel, ABC):

    def __init__(self, clf, n_folds, precision, get_delta, break_delta, predict_proba=None):

        ThresholdModel.__init__(self, clf, n_folds, predict_proba=predict_proba)
        self.precision = precision
        self._get_delta = get_delta
        self._break_delta = break_delta

    def fit(self, X, y):
        self.Y, Y_cts = np.unique(y, return_counts=True)
        if len(self.Y) > 2:
            return ValueError(
                "Threshold selection methods only works for binary quantification. Multiclass "
                "quantification is possible via OVRQuantifier class, but not recommended due to "
                "theoretical issues with that approach."
            )

        n_folds = min(self.n_folds, min(Y_cts))
        y_scores = self._cv_score(X, y, n_folds)

        if self.precision is None:
            thresholds = np.unique(y_scores)
        else:
            thresholds = np.unique(np.around(y_scores, decimals=self.precision))

        # re-sort for faster CM construction
        ind = np.argsort(y_scores)
        y_scores = y_scores[ind]
        y = y[ind]

        n_thresholds = len(thresholds)
        n = y.shape[0]

        # simulate first iteration
        t = thresholds[0]

        y_pred = [self.Y[0] if v < t else self.Y[1] for v in y_scores]

        n_tp = np.sum((y == self.Y[1]) & (y_pred == self.Y[1]))
        n_fp = np.sum((y == self.Y[0]) & (y_pred == self.Y[1]))
        self.tpr, self.fpr = n_tp / Y_cts[1], n_fp / Y_cts[0]
        delta_min = self._get_delta(self.tpr, self.fpr)
        self.threshold = t

        ir: int = 0
        while ir < n and y_scores[ir] < t:
            ir += 1

        il: int = ir

        for i in range(1, n_thresholds):

            t = thresholds[i]

            while ir < n and y_scores[ir] < t:
                ir += 1
            if il == ir:
                continue

            d = ir - il

            del_tp = np.sum(y[il:ir] == self.Y[1])

            n_tp -= del_tp
            n_fp -= d - del_tp

            tpr, fpr = n_tp / Y_cts[1], n_fp / Y_cts[0]

            delta = self._get_delta(tpr, fpr)

            if delta < delta_min:
                self.tpr, self.fpr = tpr, fpr
                self.threshold = t
                delta_min = delta

                if self._break_delta(delta):
                    break
            il = ir

        return self

