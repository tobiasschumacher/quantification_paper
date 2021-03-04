  # quantifier base class
from abc import ABC, abstractmethod
from sklearn import svm, linear_model, model_selection
from .generals import rel_target_prevalences
import numpy as np


# abstract parent class for quantifiers
# -> we need fit and predict, for similar syntax as in sklearn
class Quantifier(ABC):

    def __init__(self):
        # Y: target class labels
        self.Y = None
    #
    # @abstractmethod
    # def fit(self, X, y, Y_cts):
    #     pass

    @abstractmethod
    def predict(self, X):
        pass


# parent class for quantifiers which need classifiers to build their predictions
class CLFQuantifier(Quantifier, ABC):

    def __init__(self, clf=None, nfolds=10):
        Quantifier.__init__(self)
        if clf is None:
            self.clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
        else:
            if not hasattr(clf, "fit") or not hasattr(clf, "predict"):
                raise TypeError("Input clf needs to be a classifier with fit() and predict() function")
            self.clf = clf
        self.nfolds = nfolds
        self._clf_type = None
        self._clf_score = None

    # private functions to enable scoring with either predict_proba or decision_function of underlying clf
    # def _score_proba(self, X):
    #     return self.clf.predict_proba(X)[:,1]

    # intrinsic cv_score-function which perform CV for binary cross validation
    # in multiclass case (cf AC models)
    def _cv_score(self, X, y, nfolds):

        if not self._clf_type == "prob": #TODO: decision_function-Formate checken
            y_scores = np.zeros(y.shape)
        else:
            y_scores = np.zeros((len(y), len(self.Y)))

        if nfolds > 1:
            # estimate confusion matrix via stratified cross-validation
            skf = model_selection.StratifiedKFold(n_splits=nfolds)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.clf.fit(X_train, y_train)
                y_scores[test_index] = self._clf_score(X_test)

        # now fit real classifier
        self.clf.fit(X, y)

        if nfolds < 2:
            y_scores = self._clf_score(X)

        return y_scores

    # @abstractmethod
    # def fit(self, X, y, Y_cts):
    #     pass

    @abstractmethod
    def predict(self, X):
        pass


# parent class for all quantifiers which are built on crisp predictions of a classifier
class CrispCLFQuantifier(CLFQuantifier, ABC):

    def __init__(self, clf=None, nfolds=10):
        CLFQuantifier.__init__(self, clf)
        self.nfolds = nfolds
        self._clf_type = "crisp"
        self._clf_score = self.clf.predict
        # self.CM = None

    # def fit(self, X, y, Y_cts=None,  *args):
    #
    #     if Y_cts is None:
    #         Y_cts = np.unique(y, return_counts=True)
    #
    #     self.Y = Y_cts[0]
    #     Y_cts = Y_cts[1]
    #
    #     nfolds = min(self.nfolds, min(Y_cts))
    #     y_scores = self._cv_score(X, y, nfolds)
    #
    #     CM = metrics.confusion_matrix(y_scores, y, self.Y)
    #     self.CM = CM / CM.sum(axis=0, keepdims=True)


# parent class for all quantifiers which are built on probabilistic confidence scores of a classifier
class ProbCLFQuantifier(CLFQuantifier, ABC):

    def __init__(self, clf=None, nfolds=10):
        CLFQuantifier.__init__(self, clf=clf)
        if not hasattr(self.clf, "predict_proba"):
            raise TypeError("Input clf needs to be a classifier with predict_proba() function")
        self.nfolds = nfolds
        self._clf_type = "prob"
        self._clf_score = self.clf.predict_proba
        # self.CM = None

    # default fit according to PAC
    # def fit(self, X, y, Y_cts=None,  *args):
    #
    #     if Y_cts is None:
    #         Y_cts = np.unique(y, return_counts=True)
    #     self.Y = Y_cts[0]
    #     Y_cts = Y_cts[1]
    #
    #     nfolds = min(self.nfolds, min(Y_cts))
    #     y_scores = self._cv_score(X, y, nfolds)
    #
    #     self.CM = np.hstack([y_scores[np.where(y == l)[0]].sum(axis=0) for l in self.Y]) / Y_cts


# parent class for all quantifiers which are built on more general decision function confidence scores of a classifier
# -> can be decision functions as in SVMs, or probability scores as in logistic regression
class ScoreCLFQuantifier(CLFQuantifier, ABC):

    def __init__(self, clf=None, nfolds=10):
        CLFQuantifier.__init__(self, clf=clf)

        if not hasattr(self.clf, "decision_function"):
            self._clf_type = "prob"
            if not hasattr(self.clf, "predict_proba"):
                raise TypeError("Input clf needs to be a classifier with either decision_function() or predict_proba() "
                                "method.")
            self._clf_score = self.clf.predict_proba
        else:
            self._clf_type = "score"
            self._clf_score = self.clf.decision_function
        self.nfolds = nfolds
        # self.CM = None


