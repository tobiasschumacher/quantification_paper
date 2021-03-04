from ._base import *
import pandas as pd


# TODO: Adaption für readme
# TODO: Konzept für binning
########################################################################################################################
# parent class for Mixture Models that do not use classifiers and instead only work with features
########################################################################################################################

class FeatureModel(DMMBase, ABC):

    def __init__(self, dist, solve_cvx):
        DMMBase.__init__(self, dist=dist, solve_cvx=solve_cvx)
        self.L = None
        self.D = None
        self.X_space = None

    @abstractmethod
    def _fit_cm(self,  X, y, Y_idx, Y_cts):
        pass

    def _fit(self, X, y, Y_cts):
        self.Y = Y_cts[0]
        self.L = len(self.Y)
        Y_cts = Y_cts[1]
        Y_idx = [np.where(y == l) for l in self.Y]
        self.D = len(self.Y)

        if self.L < 2:
            raise ValueError("There is only one unique value in target vector y.")

        self._fit_cm(X, y, Y_idx, Y_cts)

    def fit(self, X, y):
        Y_cts = np.unique(y, return_counts=True)
        self._fit(X, y, Y_cts)
        return self


class ColumnProjector(FeatureModel):

    def __init__(self, dist, solve_cvx):
        FeatureModel.__init__(self, dist=dist, solve_cvx=solve_cvx)

    # TODO: Crosstab ersetzen
    def _fit_cm(self, X, y, Y_idx, Y_cts):
        self.X_space = [np.unique(X[:, j]) for j in range(self.D)]
        self.CM = np.vstack([(pd.crosstab(X[:, j], y).values / Y_cts) for j in range(self.D)])

    def score(self, X):
        n = X.shape[0]
        counts_list = [np.array([np.count_nonzero(X[:, j] == i) for i in self.X_space[j]]) for j in
                       range(self.D)]
        return np.hstack([1.0 / n * counts_list[j] for j in range(self.D)]).T


# slightly modify feature model for HDx
class HDx(ColumnProjector):

    def __init__(self, solve_cvx=True):
        ColumnProjector.__init__(self, dist="HD", solve_cvx=solve_cvx)


class RMBase(FeatureModel):

    def __init__(self, dist="L2", solve_cvx=True):
        FeatureModel.__init__(self, dist=dist, solve_cvx=solve_cvx)

    @staticmethod
    def _search_row(v, M, start_ind=0):

        j = 0
        m = M.shape[1]
        n = M.shape[0]

        il = start_ind
        ir = n

        while il + 1 < ir and j < m:
            i_tmp = np.searchsorted(a=M[il:ir, j], v=v[j], side='left')
            il += i_tmp
            i_tmp = np.searchsorted(a=M[il:ir, j], v=v[j], side='right')

            ir = il + i_tmp
            j += 1

        if il < n and np.array_equal(v, M[il, :]):
            return il
        else:
            return None

    def _fit_cm(self, X, y, Y_idx, Y_cts):

        self.D = X.shape[1]
        self.X_space = np.unique(X, axis=0)
        self.CM = np.zeros((self.X_space.shape[0], len(self.Y)))

        y_map = dict([(self.Y[i], i) for i in range(len(self.Y))])
        for i in range(len(y)):
            ind_tmp = self._search_row(X[i, :], self.X_space)
            self.CM[ind_tmp, y_map[y[i]]] += 1

        self.CM = self.CM / Y_cts

    def score(self, X):

        # sort X to accelerate later searches
        lex_ind = np.lexsort(np.rot90(X))
        X = X[lex_ind]

        xp = np.zeros(self.X_space.shape[0])

        il = 0
        for i in range(X.shape[0]):
            ind_tmp = self._search_row(X[i, :], self.X_space, il)

            if ind_tmp is None:
                continue
            il = ind_tmp
            xp[ind_tmp] += 1

        return xp * 1.0 / X.shape[0]


class ReadMe(Quantifier):

    def __init__(self, dist="L2", solve_cvx=True, n_features=None, n_subsets=100):
        self.D = None
        self.L = None
        self.n_features = n_features
        self.n_subsets = n_subsets
        self.feature_list = []
        self.quantifiers = []
        self.dist = dist
        self.solve_cvx = solve_cvx

    def fit(self, X, y):
        # build conditional probability matrix on whole feature space
        self.D = X.shape[1]

        Y_cts = np.unique(y, return_counts=True)
        self.Y = Y_cts[0]
        self.L = len(self.Y)

        if self.n_features is None:
            self.n_features = self.D.bit_length() if self.D > 25 else max(int(self.D/5), 2)

        for _ in range(self.n_subsets):
            curr_feats = np.random.choice(range(self.D), self.n_features, replace=False)
            self.feature_list.append(curr_feats)

            curr_qf = RMBase(self.dist, self.solve_cvx)
            curr_qf._fit(X[:, curr_feats], y, Y_cts)

            self.quantifiers.append(curr_qf)

        return self

    def predict(self, X):

        p = np.zeros(self.L)

        for i in range(self.n_subsets):
            p += self.quantifiers[i].predict(X[:, self.feature_list[i]])

        return p/self.n_subsets
