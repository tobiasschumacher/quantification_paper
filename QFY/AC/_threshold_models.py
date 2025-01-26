import numpy as np
from sklearn import svm

from QFY.base import ScoreCLFQuantifier
from ._base import ThresholdModel, ThresholdSelector


# ----------------------------------------------------------------------------------------------------------------------
# helper functions for threshold selectors
# ----------------------------------------------------------------------------------------------------------------------

def _rates_max(tpr, fpr):
    return fpr - tpr


def _rates_50(tpr, fpr):
    return abs(tpr - 0.5)


def _rates_x(tpr, fpr):
    return abs(tpr - 1 + fpr)


def _delta0(delta):
    return delta == 0


def _delta_false(delta):
    return False


########################################################################################################################
# ThresholdModel/Selector Classes and Functions
########################################################################################################################

class TSMax(ThresholdSelector):

    def __init__(
            self,
            clf=svm.SVC(),
            n_folds=10,
            precision=3,
            predict_proba=None
    ):
        ThresholdSelector.__init__(self, clf=clf, n_folds=n_folds, precision=precision, get_delta=_rates_max,
                                   break_delta=_delta_false, predict_proba=predict_proba)


class TSX(ThresholdSelector):

    def __init__(
            self,
            clf=svm.SVC(),
            n_folds=10,
            precision=3,
            predict_proba=None
    ):
        ThresholdSelector.__init__(self, clf=clf, n_folds=n_folds, precision=precision, get_delta=_rates_x,
                                   break_delta=_delta0, predict_proba=predict_proba)


class TS50(ThresholdSelector):

    def __init__(
            self,
            clf=svm.SVC(),
            n_folds=10,
            precision=3,
            predict_proba=None
    ):
        ThresholdSelector.__init__(self, clf=clf, n_folds=n_folds, precision=precision, get_delta=_rates_50,
                                   break_delta=_delta0, predict_proba=predict_proba)


class MS(ScoreCLFQuantifier):

    def __init__(
            self,
            clf=svm.SVC(),
            n_folds=10,
            precision=3,
            delta_min=0.25,
            predict_proba=None
    ):
        ScoreCLFQuantifier.__init__(self, clf=clf, n_folds=n_folds, predict_proba=predict_proba)
        self.precision = precision
        self.threshold_models = []
        self.delta_min = delta_min

    def fit(self, X, y):

        self.Y, Y_cts = np.unique(y, return_counts=True)
        if len(self.Y) > 2:
            return ValueError("MS only works for binary quantification. Multiclass is possible via OVRQuantifier"
                              "class, but not recommended due to theoretical issues with that approach.")
        n_folds = min(self.n_folds, min(Y_cts))

        y_scores = self._cv_score(X, y, n_folds)

        # re-sort for faster CM construction
        ind = np.argsort(y_scores)
        y_scores = y_scores[ind]
        y = y[ind]
        thresholds = np.unique(np.around(y_scores, decimals=self.precision))

        n_thresholds = len(thresholds)
        n = y.shape[0]

        # simulate first iteration
        t = thresholds[0]

        y_pred = [self.Y[0] if v < t else self.Y[1] for v in y_scores]

        n_tp = np.sum((y == self.Y[1]) & (y_pred == self.Y[1]))
        n_fp = np.sum((y == self.Y[0]) & (y_pred == self.Y[1]))
        tpr, fpr = n_tp / Y_cts[1], n_fp / Y_cts[0]

        self.threshold_models.append((ThresholdModel(self.clf, self.n_folds, tpr=tpr, fpr=fpr, threshold=t)))

        ir = 0
        while ir < n and y_scores[ir] < t:
            ir += 1

        il = ir

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

            self.threshold_models.append((ThresholdModel(self.clf, self.n_folds, tpr=tpr, fpr=fpr, threshold=t)))

            il = ir

        return self

    def predict(self, X):

        y_scores = self._clf_score(X)
        n_thresholds = len(self.threshold_models)
        p = np.zeros(n_thresholds)
        i_p = 0

        ind = np.argsort(y_scores)
        y_scores = y_scores[ind]
        n = len(y_scores)

        y_pred = np.array([self.Y[1]] * n)
        delta_max = -2

        ir, il = 0, 0

        for i in range(n_thresholds):

            curr_qf = self.threshold_models[i]
            t = curr_qf.threshold

            while ir < n and y_scores[ir] < t:
                ir += 1

            y_pred[il:ir] = self.Y[0]
            il = ir

            # filter out those adaptations where denominator would be <0.25
            delta = curr_qf.tpr - curr_qf.fpr
            if delta > self.delta_min:
                p[i_p] = ((n - ir) / n - curr_qf.fpr) / delta
                i_p += 1
            elif delta > delta_max and i_p == 0:
                if delta == 0:
                    p_max = curr_qf.tpr
                else:
                    p_max = ((n - ir) / n - curr_qf.fpr) / delta
                delta_max = delta

        if i_p < 1:
            p = np.clip(p_max, a_min=0, a_max=1)
            return np.array([1.0 - p, p])

        p = p[0:i_p]

        p = np.clip(np.median(p), a_min=0, a_max=1)

        return np.array([1.0 - p, p])
