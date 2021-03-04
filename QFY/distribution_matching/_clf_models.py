# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:44:23 2018

@author: tobi_
"""

from ._base import *
from sklearn.metrics import confusion_matrix


########################################################################################################################
# ABSTRACT BASE CLASS FOR CLASSIFIER-BASED QUANTIFIERS
########################################################################################################################

class CLFModel(DMMBase, CLFQuantifier, ABC):

    @abstractmethod
    def _fit_cm(self, y, y_scores, Y_cts):
        pass

    def _fit(self, X, y, Y_cts):
        self.Y = Y_cts[0]
        Y_cts = Y_cts[1]
        nfolds = min(self.nfolds, min(Y_cts))
        y_scores = self._cv_score(X, y, nfolds)
        self._fit_cm(y, y_scores, Y_cts)
        return self

    def fit(self, X, y):
        Y_cts = np.unique(y, return_counts=True)
        self._fit(X, y, Y_cts)
        return self


########################################################################################################################
# CLASS FOR DyS QUANTIFIERS
# -> quantifies based on BINNED Classifier Scores
########################################################################################################################

class BinaryDyS(CLFModel, ScoreCLFQuantifier):

    def __init__(self, clf, distance, nbins, nfolds, solve_cvx):
        ScoreCLFQuantifier.__init__(self, clf=clf, nfolds=nfolds)
        DMMBase.__init__(self, dist=distance, solve_cvx=solve_cvx)
        self.score_range = None
        self.nbins = nbins

    def _fit_cm(self, y, y_scores, Y_cts):

        if self._clf_type == 'prob':
            self.score_range = (0, 1)
        else:
            self.score_range = (np.min(y_scores), np.max(y_scores))

        self.CM = np.vstack([np.histogram(y_scores[np.where(y == l)[0]], bins=self.nbins, range=self.score_range)[0]
                             for l in self.Y]).T / Y_cts

    def score(self, X):
        if self._clf_type == 'prob':
            y_scores = self.clf.predict_proba(X)[:, -1]
            yp, _ = np.histogram(y_scores, bins=self.nbins, range=self.score_range)
        else:
            y_scores = self.clf.decision_function(X)
            yp, _ = np.histogram(y_scores, bins=self.nbins, range=self.score_range)
            yp[0] += np.sum(y_scores < self.score_range[0])
            yp[-1] += np.sum(y_scores > self.score_range[1])

        return yp / X.shape[0]


########################################################################################################################
# Block of final quantifiers
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Default CLassifier-Based Mixture Model
# -> Variant of AC
# ----------------------------------------------------------------------------------------------------------------------
class GAC(CLFModel, CrispCLFQuantifier):

    def __init__(self, clf=svm.LinearSVC(), distance="L2", nfolds=10, solve_cvx=True):
        CrispCLFQuantifier.__init__(self, clf=clf, nfolds=nfolds)
        DMMBase.__init__(self, dist=distance, solve_cvx=solve_cvx)

    def _fit_cm(self, y, y_scores, Y_cts):
        CM = confusion_matrix(y, y_scores, self.Y).transpose()
        self.CM = CM / Y_cts

    def score(self, X):
        y = self.clf.predict(X)
        return rel_target_prevalences(y, Y=self.Y)


# ----------------------------------------------------------------------------------------------------------------------
# Default Probabilistic CLassifier-Based Mixture Model
# -> Variant of PAC
# ----------------------------------------------------------------------------------------------------------------------
class GPAC(CLFModel, ProbCLFQuantifier):

    def __init__(self,
                 clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'),
                 distance="L2",
                 nfolds=10,
                 solve_cvx=True):

        ProbCLFQuantifier.__init__(self, clf=clf, nfolds=nfolds)
        DMMBase.__init__(self, dist=distance, solve_cvx=solve_cvx)

    def _fit_cm(self, y, y_scores, Y_cts):
        CM = np.zeros(shape=(len(self.Y), len(self.Y)))
        for l in range(len(self.Y)):
            idx = np.where(y == self.Y[l])[0]
            CM[:, l] += y_scores[idx].sum(axis=0)

        self.CM = CM / Y_cts

    def score(self, X):
        return self.clf.predict_proba(X).sum(axis=0) * 1.0 / X.shape[0]


# ----------------------------------------------------------------------------------------------------------------------
# Friedman's method
# ----------------------------------------------------------------------------------------------------------------------
class FM(CLFModel, ProbCLFQuantifier):

    def __init__(self,
                 clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'),
                 distance="L2",
                 nfolds=10,
                 solve_cvx=True):

        ProbCLFQuantifier.__init__(self, clf=clf, nfolds=nfolds)
        DMMBase.__init__(self, dist=distance, solve_cvx=solve_cvx)

        self.y_prevs = None

    def _fit_cm(self, y, y_scores, Y_cts):
        self.y_prevs = Y_cts / len(y)

        CM = np.zeros(shape=(len(self.Y), len(self.Y)))
        for l in range(len(self.Y)):
            idx = np.where(y == self.Y[l])[0]
            CM[:, l] += (y_scores[idx] > self.y_prevs).sum(axis=0)

        self.CM = CM / Y_cts

    def score(self, X):
        return np.sum(self.clf.predict_proba(X) > self.y_prevs, axis=0) / X.shape[0]


# ----------------------------------------------------------------------------------------------------------------------
# Hellinger distance quantifiers
# ----------------------------------------------------------------------------------------------------------------------

# HDy simply combination of parents
class HDy(GAC):

    def __init__(self, clf=svm.SVC(), nfolds=10):
        GAC.__init__(self, clf=clf, distance="HD", nfolds=nfolds, solve_cvx=True)

# TODO: ggf mehr distanzen einbinden