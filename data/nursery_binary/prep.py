
import pandas as pd


def prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"

    colnames = ["att" + str(i + 1) for i in range(9)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta.att9 = dta.att9.replace({"not_recom": 0, "recommend": 1, "very_recom": 1, "priority": 1, "spec_prior": 1})
    dta = pd.get_dummies(dta)

    return dta

# dta.to_pickle("dta.pkl")
