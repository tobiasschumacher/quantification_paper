# quantifier base class
from abc import ABC, abstractmethod
from sklearn import model_selection
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

    def __init__(self, clf=None, n_folds=10):
        Quantifier.__init__(self)
        if not hasattr(clf, "fit") or not hasattr(clf, "predict"):
            raise TypeError("Input clf needs to be a classifier with fit() and predict() function")
        self.clf = clf
        self.n_folds = n_folds
        self._init_scores = None
        self._clf_score = None

    # intrinsic cv_score-function which performs CV for binary cross validation
    # in multiclass case (cf AC models)

    def _cv_score(self, X, y, n_folds):

        y_scores = self._init_scores(y)

        if n_folds > 1:

            # estimate confusion matrix via stratified cross-validation
            skf = model_selection.StratifiedKFold(n_splits=n_folds)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.clf.fit(X_train, y_train)
                y_scores[test_index] = self._clf_score(X_test)

        # now fit real classifier
        self.clf.fit(X, y)

        if n_folds < 2:
            y_scores = self._clf_score(X)

        return y_scores

    @abstractmethod
    def predict(self, X):
        pass


# parent class for all quantifiers which are built on crisp predictions of a classifier
class CrispCLFQuantifier(CLFQuantifier, ABC):

    def __init__(self, clf=None, n_folds=10):
        CLFQuantifier.__init__(self, clf)
        self.n_folds = n_folds
        self._init_scores = lambda y: np.zeros(y.shape)
        self._clf_score = self.clf.predict


# parent class for all quantifiers which are built on probabilistic confidence scores of a classifier
class ProbCLFQuantifier(CLFQuantifier, ABC):

    def _init_prob_matrix(self, y):
        return np.zeros((y.shape[0], len(self.Y)))

    def __init__(self, clf=None, n_folds=10):
        CLFQuantifier.__init__(self, clf=clf)
        if not hasattr(self.clf, "predict_proba"):
            raise TypeError("Input clf needs to be a classifier with predict_proba() function")
        self.n_folds = n_folds
        self._init_scores = self._init_prob_matrix
        self._clf_score = self.clf.predict_proba


# parent class for all quantifiers which are built on more general decision function confidence scores of a classifier
# -> can be decision functions as in SVMs, or probability scores as in logistic regression
class ScoreCLFQuantifier(CLFQuantifier, ABC):

    def _score_proba(self, X):
        return self.clf.predict_proba(X)[:, -1]

    def __init__(self, clf=None, n_folds=10, predict_proba=None):
        CLFQuantifier.__init__(self, clf=clf)

        self.predict_proba = False if predict_proba is None else predict_proba

        if self.predict_proba:
            if not hasattr(self.clf, "predict_proba"):
                raise TypeError("Input clf needs to be a classifier with predict_proba() method if predict_proba is "
                                "set to True.")
            self._clf_score = self._score_proba
        elif not hasattr(self.clf, "decision_function"):
            if not hasattr(self.clf, "predict_proba"):
                raise TypeError("Input clf needs to be a classifier with either decision_function() or predict_proba() "
                                "method.")
            self._clf_score = self._score_proba
        else:
            self._clf_score = self.clf.decision_function

        self.n_folds = n_folds
        self._init_scores = lambda y: np.zeros(y.shape)
