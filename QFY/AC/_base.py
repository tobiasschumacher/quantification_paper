from abc import ABC, abstractmethod
import numpy as np
from QFY.base import CLFQuantifier, CrispCLFQuantifier, ScoreCLFQuantifier, ProbCLFQuantifier


########################################################################################################################
# (P)AC Classes
########################################################################################################################

# AC Base class
class ACModel(CLFQuantifier, ABC):

    def __init__(self):
        self.tpr, self.fpr = None, None

    @abstractmethod
    def _get_rates(self, y, y_scores, Y_cts):
        pass

    @abstractmethod
    def _score_pos(self, X):
        pass

    def fit(self, X, y, Y_cts):

        self.Y = Y_cts[0]
        Y_cts = Y_cts[1]

        nfolds = min(self.nfolds, min(Y_cts))
        y_scores = self._cv_score(X, y, nfolds)

        self.tpr, self.fpr = self._get_rates(y, y_scores, Y_cts)

        return self

    def predict(self, X):

        y_pos = self._score_pos(X)
        delta = self.tpr - self.fpr

        if delta == 0:
            return y_pos

        p = (y_pos - self.fpr) / delta

        return min(1.0, max(0.0, p))


class BinaryAC(ACModel, CrispCLFQuantifier):

    def __init__(self, clf, nfolds):
        ACModel.__init__(self)
        CrispCLFQuantifier.__init__(self, clf, nfolds)

    def _score_pos(self, X):
        return np.sum(self.clf.predict(X) == self.Y[1]) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pos_ind = y == self.Y[1]
        tpr = np.sum(y_scores[pos_ind]) / Y_cts[1]
        fpr = np.sum(y_scores[~pos_ind]) / Y_cts[0]
        return tpr, fpr


class BinaryPAC(ACModel, ProbCLFQuantifier):

    def __init__(self, clf, nfolds):
        ACModel.__init__(self)
        ProbCLFQuantifier.__init__(self, clf, nfolds)

    def _score_pos(self, X):
        return np.sum(self._clf_score(X)[:, 1]) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pos_ind = y == self.Y[1]
        tpr = np.sum(y_scores[pos_ind,1]) / Y_cts[1]
        fpr = np.sum(y_scores[~pos_ind,1]) / Y_cts[0]
        return tpr, fpr


########################################################################################################################
# ThresholdModel/Selector Classes and Functions
########################################################################################################################


##
# helper functions for threshold selectors
##
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


class ThresholdModel(ACModel, ScoreCLFQuantifier):

    def __init__(self, clf, nfolds, tpr=None, fpr=None, threshold=None):
        ACModel.__init__(self)
        ScoreCLFQuantifier.__init__(self, clf=clf, nfolds=nfolds)

        self.threshold = threshold
        self.tpr, self.fpr = tpr, fpr

    def _score_pos(self, X):
        return np.sum(self._clf_score(X) >= self.threshold) / X.shape[0]

    def _get_rates(self, y, y_scores, Y_cts):
        pass


# Threshold Selector Base Class
class ThresholdSelector(ThresholdModel, ABC):

    # TODO: check rates in init
    def __init__(self, clf, nfolds, precision, get_delta, break_delta):

        ThresholdModel.__init__(self, clf, nfolds)
        self.precision = precision
        self._get_delta = get_delta
        self._break_delta = break_delta

    def fit(self, X, y, Y_cts):

        self.Y = Y_cts[0]
        Y_cts = Y_cts[1]

        nfolds = min(self.nfolds, min(Y_cts))
        y_scores = self._cv_score(X, y, nfolds)
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
        while y_scores[ir] < t:
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


class BinaryTSMax(ThresholdSelector):

    def __init__(self, clf, nfolds=10, precision=2):
        ThresholdSelector.__init__(self, clf=clf, nfolds=nfolds, precision=precision, get_delta=_rates_max,
                                   break_delta=_delta_false)


class BinaryTSX(ThresholdSelector):

    def __init__(self, clf, nfolds, precision):
        ThresholdSelector.__init__(self, clf=clf, nfolds=nfolds, precision=precision, get_delta=_rates_x,
                                   break_delta=_delta0)


class BinaryTS50(ThresholdSelector):

    def __init__(self, clf, nfolds=10, precision=2):
        ThresholdSelector.__init__(self, clf=clf, nfolds=nfolds, precision=precision, get_delta=_rates_50,
                                   break_delta=_delta0)


class BinaryMS(ScoreCLFQuantifier):

    def __init__(self, clf, nfolds, precision, delta_min):
        ScoreCLFQuantifier.__init__(self, clf=clf, nfolds=nfolds)
        self.precision = precision
        self.threshold_models = []
        self.delta_min = delta_min

    def fit(self, X, y, Y_cts):

        self.Y = Y_cts[0]
        Y_cts = Y_cts[1]
        nfolds = min(self.nfolds, min(Y_cts))

        y_scores = self._cv_score(X, y, nfolds)

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

        self.threshold_models.append((ThresholdModel(self.clf, self.nfolds, tpr=tpr, fpr=fpr, threshold=t)))

        ir = 0
        while y_scores[ir] < t:
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

            self.threshold_models.append((ThresholdModel(self.clf, self.nfolds, tpr=tpr, fpr=fpr, threshold=t)))

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
            return min(1, max(0, p_max))

        p = p[0:i_p]

        return max(0.0, (min(1.0, np.median(p))))
