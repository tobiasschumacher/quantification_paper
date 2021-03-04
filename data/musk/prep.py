# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "clean2.data"

  colnames = ["att" + str(i-1) for i in range(169)]

  dta = pd.read_csv(url,
              header = None,
              names = colnames,
              skipinitialspace = True)

  dta = dta.drop(["att-1","att0"], axis = 1)
  
  dta -= dta.min()
  dta /= dta.max()
  dta["att167"] = dta["att167"].astype("int64")
  
  
  if binned:
    for col in list(dta)[:-1]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
