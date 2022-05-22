
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data(binned=False):
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00315/Geographical%20Original%20of%20Music.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[6]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=None,
                      names=["att" + str(i + 1) for i in range(118)],
                      skipinitialspace=True)

    dta = dta.drop("att118", axis=1)

    bins = [-40, 35, 60]
    labels = [0, 1]
    dta['att117'] = pd.cut(dta['att117'], bins=bins, labels=labels)
    dta['att117'] = dta['att117'].astype("int64")

    if binned:
        for col in list(dta)[:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
