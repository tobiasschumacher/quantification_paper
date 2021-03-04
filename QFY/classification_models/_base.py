from ..base import CLFQuantifier, ProbCLFQuantifier, rel_target_prevalences
import numpy as np
from sklearn import linear_model, svm


########################################################################################################################
# CC/PCC Classes
########################################################################################################################

class CC(CLFQuantifier):

    def __init__(self, clf=svm.SVC()):
        CLFQuantifier.__init__(self, clf=clf, nfolds=0)

    def fit(self, X, y):
        self.Y = np.unique(y)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        y = self.clf.predict(X)
        return rel_target_prevalences(y, self.Y)


class PCC(ProbCLFQuantifier, CC):

    def __init__(self, clf=linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')):
        ProbCLFQuantifier.__init__(self, clf=clf, nfolds=0)

    def predict(self, X):
        yp = self.clf.predict_proba(X)
        return 1.0 / X.shape[0] * sum(yp)
