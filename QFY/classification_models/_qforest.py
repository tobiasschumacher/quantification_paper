import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from QFY.base import Quantifier


class QuantificationForest(Quantifier):

    def __init__(self, qforest_path, id_str = None, n_trees=100, seed=None):

        Quantifier.__init__(self)
        self.qforest_path = qforest_path
        assert Path(
            qforest_path).is_dir(), "The directory of the quantification tree executables weka.jar and quantify.jar " \
                                    "needs to be stored in QFOREST_PATH. These executables were kindly provided to " \
                                    "the authors of this work by Ms. Letizia Milli (letiziam83@gmail.com). Please ask" \
                                    "her for these corresponding executables. "

        self.model_id = id_str if id_str is not None else "qforest"
        self.training_seed = seed
        self.n_trees = n_trees
        self.sep = ';' if sys.platform.startswith('win32') else ':'

    @staticmethod
    def _parse_binary_log(log):
        stats = log.stdout.decode('utf-8').split('=== Confusion Matrix ===\n\n')[1].split('\n')

        cc_stats = stats[4:6]
        cc_stats = [s.split("%")[0] for s in cc_stats]
        cc_stats = np.array([float(s.split(":")[1]) / 100 for s in cc_stats])

        ac_stats = stats[7:9]
        ac_stats = [s.split("%")[0] for s in ac_stats]
        ac_stats = np.array([float(s.split(":")[1]) / 100 for s in ac_stats])

        return cc_stats, ac_stats

    @staticmethod
    def _parse_multiclass_log(log, n_classes):
        stats = log.stdout.decode('utf-8').split('=== Confusion Matrix ===\n\n')[1].split('\n')

        cc_idx = 2 + n_classes

        cc_stats = stats[cc_idx:cc_idx + n_classes]
        cc_stats = [s.split("%")[0] for s in cc_stats]
        cc_stats = np.array([float(s.split(":")[1]) / 100 for s in cc_stats])

        return cc_stats

    @staticmethod
    def _pandas2arff(df, fname, relname="data"):
        def convert_dtype(i):

            dtype = str(dtypes[i])

            if dtype == "category":
                cats = df.iloc[:, i].cat.categories
                return "{" + ','.join(str(c) for c in cats) + "}"
            else:
                return {"int64": "numeric",
                        "float64": "numeric",
                        }.get(dtype, "string")

        csv_str = df.to_csv(header=False, index=False, line_terminator="\n")

        cols = list(df)
        cols = [attr.replace(" ", "_") for attr in cols]
        cols = [attr.replace("'", "") for attr in cols]
        dtypes = [str(d) for d in df.dtypes]
        attr_list = ["@attribute " + str(cols[i]) + " " + convert_dtype(i) for i in range(len(cols))]
        attr_str = "\n".join(attr_list)

        arff_str = "@relation " + str(relname) + "\n" + attr_str + "\n@data\n" + csv_str

        if fname[-5:] != ".arff":
            fname += ".arff"

        arff_file = open(fname, "w")
        arff_file.write(arff_str)
        arff_file.close()

    def fit(self, X, y):

        self.Y = np.unique(y)

        train_cmd = f"java -Xmx1G -cp quantify.jar{self.sep}weka.jar{self.sep}. " \
                    f"weka.classifiers.trees.RandomForest -I {self.n_trees} " \
                    f"-t train_{self.model_id}.arff -d model_{self.model_id}"

        if self.training_seed is not None:
            train_cmd += f" -S {self.training_seed}"

        print(train_cmd)

        train_data = pd.DataFrame(X, columns=[f"col{i}" for i in range(1, X.shape[1]+1)])
        train_data = train_data.assign(y=pd.Series(y).values)
        train_data["y"] = train_data["y"].astype("category")
        self._pandas2arff(train_data, f"{self.qforest_path}train_{self.model_id}.arff")

        log = subprocess.run(train_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             cwd=self.qforest_path).stderr.decode('utf-8')
        if len(log) > 1:
            print(log)
            raise ValueError(log)

    def predict(self, X, y=None, return_ac=True):

        test_cmd = f"java  -Xmx1G -cp quantify.jar{self.sep}weka.jar{self.sep}. weka.classifiers.trees.RandomForest " \
                   f" -l  model_{self.model_id} -T test_{self.model_id}.arff"

        # label vector y is needed as part of test data, although it is just used to compute error scores, that we are
        # not interesed in at this point - we just want the prevalence estimations, that do not depend on the vector y,
        # from the returned log files. Thus, we create a dummy vector here
        if y is None:
            y = self.Y.tolist() + [self.Y[0]]*(X.shape[0]-len(self.Y))

        test_data = pd.DataFrame(X, columns=[f"col{i}" for i in range(1, X.shape[1]+1)])
        test_data = test_data.assign(y=pd.Series(y).values)
        test_data["y"] = test_data["y"].astype("category")

        self._pandas2arff(test_data, f"{self.qforest_path}test_{self.model_id}.arff")

        log = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, cwd=self.qforest_path)

        n_classes = len(self.Y)
        if n_classes > 2:
            cc_stats = self._parse_multiclass_log(log, n_classes)
        else:
            cc_stats, ac_stats = self._parse_binary_log(log)

            if return_ac:
                if np.any(np.isnan(ac_stats)):
                    ac_stats = cc_stats
                else:
                    ac_stats = np.clip(ac_stats, 0, 1)
                print("CC: " + str(cc_stats))
                print("AC: " + str(ac_stats))

                return np.concatenate([cc_stats, ac_stats])

        print("CC: " + str(cc_stats))
        return cc_stats
