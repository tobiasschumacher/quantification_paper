import math
import cvxpy as cvx
import warnings
from QFY.base import *


# GSS constants
invphi = (math.sqrt(5) - 1) / 2  # 1/phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2


########################################################################################################################
# MIXTURE MODEL PARENT CLASS
########################################################################################################################

class DMMBase(Quantifier, ABC):

    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE FUNCTIONS
    # ------------------------------------------------------------------------------------------------------------------

    # DISTANCE FUNCTIONS

    def _l1_distance(self, p, yp):
        return np.linalg.norm(self.CM.dot(p) - yp, ord=1)

    def _l2_distance(self, p, yp):
        return np.linalg.norm(self.CM.dot(p) - yp)

    @staticmethod
    def _hd_div(p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    def _hellinger_divergence(self, p, yp):
        return self._hd_div(self.CM.dot(p), yp)

    def _summed_hellinger_divergence(self, p, yp):
        return sum(self._hd_div(self.CM[i].dot(p), yp[i]) for i in range(len(self.CM)))

    def _topsoe_distance(self, p, yp):
        p = self.CM.dot(p)
        return sum(p[i] * np.log(2 * p[i] / (p[i] + yp[i])) if p[i] != 0 else 0 for i in range(p.shape[0])) + \
            sum(yp[i] * np.log(2 * yp[i] / (p[i] + yp[i])) if yp[i] != 0 else 0 for i in range(p.shape[0]))

    _distance_dict = dict({
        'L1': _l1_distance,
        'L2': _l2_distance,
        'HD': _hellinger_divergence,
        'summed_HD': _summed_hellinger_divergence,
        'TS': _topsoe_distance
    })

    # CONVEX PROBLEM SOLVER FUNCTIONS

    def _solve_l1(self, yp):
        p = cvx.Variable(self.CM.shape[1])
        constraints = [p >= 0, cvx.sum(p) == 1.0]
        problem = cvx.Problem(cvx.Minimize(cvx.norm1(self.CM @ p - yp)), constraints)
        problem.solve()
        return p.value

    def _solve_l2(self, yp):
        p = cvx.Variable(self.CM.shape[1])
        constraints = [p >= 0, cvx.sum(p) == 1.0]
        problem = cvx.Problem(cvx.Minimize(cvx.norm(self.CM @ p - yp)), constraints)
        problem.solve()
        return p.value

    def _solve_hellinger(self, yp):
        p = cvx.Variable(self.CM.shape[1])
        constraints = [p >= 0, cvx.sum(p) == 1.0]
        problem = cvx.Problem(cvx.Maximize(cvx.sum(cvx.sqrt(cvx.multiply(yp, self.CM @ p)))), constraints)
        problem.solve()
        return p.value

    def _solve_topsoe(self, yp):
        p = cvx.Variable(self.CM.shape[1])
        constraints = [p >= 0, cvx.sum(p) == 1.0]
        problem = cvx.Problem(cvx.Minimize(cvx.sum(cvx.kl_div(2 * self.CM @ p, yp) +
                                                   cvx.kl_div(2 * yp, self.CM @ p))),
                              constraints)
        problem.solve(max_iters=10000)
        return p.value

    _cvx_dict = dict({
        'L1': _solve_l1,
        'L2': _solve_l2,
        'HD': _solve_hellinger,
        'TS': _solve_topsoe
    })

    ####################################################################################################################
    # PUBLIC FUNCTIONS
    ####################################################################################################################

    def __init__(self, dist=None, solve_cvx=True):

        Quantifier.__init__(self)

        self.CM = None

        if dist is None:
            self.dist = self._l1_distance
            if solve_cvx is True:
                self._solve_cvx = self._solve_l1
            else:
                self._solve_cvx = None
        elif dist in self._distance_dict.keys():
            self.dist = self._distance_dict[dist]
            if solve_cvx is True and dist in ["L1", "L2", "HD", "TS"]:
                self._solve_cvx = self._cvx_dict[dist]
            else:
                self._solve_cvx = None
        else:
            raise ValueError("Invalid Distance Function!")

    @abstractmethod
    def score(self, X):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        yp = self.score(X)
        if self._solve_cvx is None:
            return self.gs_search(yp)
        else:
            # use cxypy to solve optimization problem
            try:
                p = self._solve_cvx(self, yp)
                if p is None:
                    warnings.warn("Empty result in convex optimization, used gs search instead")
                    return self.gs_search(yp)
                return np.array(p).squeeze()
            except cvx.SolverError:
                warnings.warn("SolverError in cvxpy, used gs search instead")
                return self.gs_search(yp)

    def gs_search(self, yp, eps=1e-04):

        (a, b) = (0.0, 1.0)
        h = 1.0

        # required steps to achieve tolerance
        n = int(math.ceil(math.log(eps / h) / math.log(invphi)))

        c = a + invphi2 * h
        d = a + invphi * h
        yc = self.dist(self, np.array([c, 1 - c]), yp)
        yd = self.dist(self, np.array([d, 1 - d]), yp)

        for k in range(n - 1):
            if yc < yd:
                b = d
                d = c
                yd = yc
                h = invphi * h
                c = a + invphi2 * h
                yc = self.dist(self, np.array([c, 1 - c]), yp)
            else:
                a = c
                c = d
                yc = yd
                h = invphi * h
                d = a + invphi * h
                yd = self.dist(self, np.array([d, 1 - d]), yp)

        if yc < yd:
            ya = self.dist(self, np.array([a, 1 - a]), yp)
            m = (a + d) / 2
            ym = self.dist(self, np.array([m, 1 - m]), yp)
            p = [a, m, d][int(np.argmin([ya, ym, yd]))]

        else:
            yb = self.dist(self, np.array([b, 1 - b]), yp)
            m = (b + c) / 2
            ym = self.dist(self, np.array([m, 1 - m]), yp)
            p = [b, m, c][int(np.argmin([yb, ym, yc]))]

        return np.array([p, 1 - p])