from os.path import join, exists
import tempfile
import subprocess
from subprocess import PIPE, STDOUT
from random import randint
import numpy as np
from sklearn.datasets import dump_svmlight_file

from ._base import CC


class SVMperfCLassifier:
    # losses with their respective codes in svm_perf implementation
    valid_losses = {'01': 0, 'kld': 12, 'nkld': 13, 'q': 22, 'qacc': 23, 'qf1': 24, 'qgm': 25}  # 12,22
    valid_kernels = {'linear': 0, 'poly': 1, 'rbf': 2, 'sig': 3}  # 0, 2

    def __init__(self, svmperf_path, kernel="linear", gamma=1, C=0.01, loss='01', timeout=None):
        assert loss in self.valid_losses, 'unsupported loss {}, valid ones are {}'.format(loss, list(
            self.valid_losses.keys()))
        assert kernel in self.valid_kernels, 'unsupported kerneö {}, valid ones are {}'.format(loss, list(
            self.valid_kernels.keys()))
        self.tmpdir = None
        self.svmperf_learn = join(svmperf_path, 'svm_perf_learn')
        self.svmperf_classify = join(svmperf_path, 'svm_perf_classify')
        self.loss = '-w 3 -l ' + str(self.valid_losses[loss])
        self.kernel = '-t ' + str(self.valid_kernels[kernel])
        if kernel == "rbf":
            self.kernel += " --b 0"
        self.gamma = '-g ' + str(gamma)
        self.param_C = '-c ' + str(C)
        self.__name__ = 'SVMperf-' + loss
        self.model = None
        self.Y = None
        self.timeout = timeout

    def fit(self, X, y):
        self.Y = np.unique(y)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model = join(self.tmpdir.name, 'model')
        print(self.model)
        traindat = join(self.tmpdir.name, 'train.dat')

        dump_svmlight_file(X, y, traindat, zero_based=False)
        cmd = ' '.join([self.svmperf_learn, self.kernel, self.param_C, self.gamma, self.loss, traindat, self.model])

        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT, timeout=self.timeout)

        print(p.stdout.decode('utf-8'))

        return self

    def predict(self, X, y=None):
        assert self.tmpdir is not None, 'predict called before fit, or model directory corrupted'
        assert exists(self.model), 'model not found'
        if y is None:
            y = np.zeros(X.shape[0])

        random_code = '-'.join(
            str(randint(0, 1000000)) for _ in range(5))
        predictions = join(self.tmpdir.name, 'predictions' + random_code + '.dat')
        testdat = join(self.tmpdir.name, 'test' + random_code + '.dat')
        dump_svmlight_file(X, y, testdat, zero_based=False)

        cmd = ' '.join([self.svmperf_classify, testdat, self.model, predictions])
        print('[Running]', cmd, "\n")
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        return [self.Y[1] if p > 0 else self.Y[0] for p in np.loadtxt(predictions)]

    def cleanup(self):
        self.tmpdir.cleanup()


class SVMPerf(CC):

    def __init__(self,
                 svmperf_path,
                 kernel="rbf",
                 C=1,
                 gamma=1,
                 loss='kld',
                 timeout=None):

        CC.__init__(self, clf=SVMperfCLassifier(svmperf_path=svmperf_path,
                                                kernel=kernel,
                                                C=C,
                                                gamma=gamma,
                                                loss=loss,
                                                timeout=timeout)
                    )

    def cleanup(self):
        self.clf.cleanup()


class SVM_KLD(SVMPerf):

    def __init__(self,
                 svmperf_path,
                 C=1,
                 timeout=None):

        SVMPerf.__init__(self,
                         svmperf_path=svmperf_path,
                         kernel="linear",
                         C=C,
                         loss='kld',
                         timeout=timeout
                         )


class SVM_Q(SVMPerf):

    def __init__(self,
                 svmperf_path,
                 C=1,
                 timeout=None):

        SVMPerf.__init__(self,
                         svmperf_path=svmperf_path,
                         kernel="linear",
                         C=C,
                         loss='q',
                         timeout=timeout
                         )


class RBF_KLD(SVMPerf):

    def __init__(self,
                 svmperf_path,
                 C=1,
                 gamma=1,
                 timeout=None):

        SVMPerf.__init__(self,
                         svmperf_path=svmperf_path,
                         kernel="rbf",
                         C=C,
                         gamma=gamma,
                         loss='kld',
                         timeout=timeout
                         )


class RBF_Q(SVMPerf):

    def __init__(self,
                 svmperf_path,
                 C=1,
                 gamma=1,
                 timeout=None):

        SVMPerf.__init__(self,
                         svmperf_path=svmperf_path,
                         kernel="rbf",
                         C=C,
                         gamma=gamma,
                         loss='q',
                         timeout=timeout
                         )
