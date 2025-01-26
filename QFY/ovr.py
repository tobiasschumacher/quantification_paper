import numpy as np

from copy import deepcopy
import warnings
from joblib import Parallel, delayed

from sklearn.preprocessing import label_binarize

from .base import Quantifier


class OVRQuantifier(Quantifier):

    def __init__(self, qf, clf_param_dict=None, n_jobs=None):
        Quantifier.__init__(self)
        self.qf = qf
        self.qf_models = {}
        self.n_jobs = n_jobs
        if clf_param_dict is not None:
            if not hasattr(qf, "clf"):
                raise ValueError("Input quantifier does not use a base classifier, "
                                 "therefore clf_param_dict cannot be applied")
            if len(clf_param_dict) < 3:
                raise ValueError("Classifier parameter dicts can only be used for multiclass problems")
            for yc, params in clf_param_dict.items():
                curr_qf = deepcopy(qf)
                curr_qf.clf.set_params(**params)
                self.qf_models[yc] = curr_qf

            self.L = len(self.qf_models)
        else:
            self.L = None

    def _fit(self, yc, X, y):
        qf = deepcopy(self.qf_models[yc])
        qf.fit(X, y)
        return qf

    def _predict(self, yc, X):
        return self.qf_models[yc].predict(X)[1]

    def fit(self, X, y):

        self.Y = np.unique(y)

        if self.L is not None:
            if self.L != len(self.Y) or set(self.Y) != set(self.qf_models.keys()):
                raise ValueError("Classes in vector y do not correspond to classes specified "
                                 "when initializing this quantifier")
        else:
            self.L = len(self.Y)

            for i, yc in enumerate(self.Y):
                self.qf_models[yc] = deepcopy(self.qf)

        y = label_binarize(y, classes=self.Y)

        if self.n_jobs is not None:
            fitted_models = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(yc, X, y[:, i])
                                                         for i, yc in enumerate(self.Y))
            for i, yc in enumerate(self.Y):
                self.qf_models[yc] = fitted_models[i]
        else:
            for i, yc in enumerate(self.Y):
                self.qf_models[yc].fit(X, y[:, i])

        return self

    def predict(self, X):

        if self.n_jobs is not None:
            p = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(yc, X) for yc in self.Y)
            p = np.array(p)
        else:
            p = np.zeros(self.L)
            for i, yc in enumerate(self.Y):
                p[i] = self.qf_models[yc].predict(X)[1]

        p_sum = np.sum(p)
        if p_sum == 0:
            warnings.warn("OVR Quantifier estimated prevalence of every class as 0. "
                          "Therefore, uniform distribution was returned.")
            return np.array([1.0 / self.L] * self.L)

        return p / p_sum
