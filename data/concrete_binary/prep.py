# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

  colnames = ["comp" + str(i+1) for i in range(8)] + ["strength"]

  dta = pd.read_excel(url,
              header = 0,
              names = colnames,
              skipinitialspace = True)
  
  bins = [0, 35, 100]
  labels = [0,1]
  dta['strength'] = pd.cut(dta['strength'], bins = bins, labels = labels)
  dta['strength'] = dta['strength'].astype("int64")

  if binned:
    for col in list(dta)[:-1]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")

