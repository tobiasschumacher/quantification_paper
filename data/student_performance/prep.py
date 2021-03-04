# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "StudentsPerformance.csv"

  dta = pd.read_csv(url,
              header = 0,
              skipinitialspace = True)
  
  dta = pd.get_dummies(dta)
  
  bins = [-1,66,100]
  labels = [0,1]
  dta['math score'] = pd.cut(dta['math score'], bins = bins, labels = labels)
  dta['math score'] = dta['math score'].astype("int64")
  
  if binned:
    for col in list(dta)[1:3]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
