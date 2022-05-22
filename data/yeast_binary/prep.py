import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

    colnames = ['name',
                'mcg',
                'gvh',
                'alm',
                'mit',
                'erl',
                'pox',
                'vac',
                'nuc',
                'class']

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col=False,
                      sep=' ',
                      skipinitialspace=True)

    dta = dta.drop(columns=['name'])
    dta['class'] = dta['class'].replace({'CYT': 1,
                                     'NUC': 0,
                                     'MIT': 0,
                                     'ME3': 0,
                                     'ME2': 0,
                                     'ME1': 0,
                                     'EXC': 0,
                                     'VAC': 0,
                                     'POX': 0,
                                     'ERL': 0})

    # dta.to_pickle("dta.pkl")

    if binned:
        bins = [0, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['mcg'] = pd.cut(dta['mcg'], bins=bins, labels=labels)
        dta['mcg'] = dta['mcg'].astype("int64")

        bins = [0, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['gvh'] = pd.cut(dta['gvh'], bins=bins, labels=labels)
        dta['gvh'] = dta['gvh'].astype("int64")

        bins = [0, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['alm'] = pd.cut(dta['alm'], bins=bins, labels=labels)
        dta['alm'] = dta['alm'].astype("int64")

        bins = [-0.1, 0.1, 0.2, 0.3, 1]
        labels = [1, 2, 3, 4]
        dta['mit'] = pd.cut(dta['mit'], bins=bins, labels=labels)
        dta['mit'] = dta['mit'].astype("int64")

        bins = [-0.1, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['pox'] = pd.cut(dta['pox'], bins=bins, labels=labels)
        dta['pox'] = dta['pox'].astype("int64")

        bins = [-0.1, 0.25, 0.35, 1]
        labels = [1, 2, 3]
        dta['vac'] = pd.cut(dta['vac'], bins=bins, labels=labels)
        dta['vac'] = dta['vac'].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta
