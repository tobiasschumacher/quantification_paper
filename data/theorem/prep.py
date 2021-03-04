# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "all-data-raw.csv"

  colnames = ["att" + str(i+1) for i in range(58)]

  dta = pd.read_csv(url,
              header = None,
              names = colnames,
              skiprows = 1,
              skipinitialspace = True)
  
  dta[dta == -100] = 1000
  dta["min_time"] = dta.iloc[:,-5:].min(axis = 1)
  dta["res"] = dta.iloc[:,-7:-2].idxmin(axis = 1)
  dta["res"] = dta["res"].apply(lambda t: int(t[-1]) - 3)
  dta.loc[dta["min_time"]> 100,"res"] = 0
  dta = dta.drop(["att54","att55","att56","att57","att58","min_time","att5","att35"], axis = 1)
  
  bins = [-1, 0, 1, 5]
  labels = [0,1,2]
  dta['res'] = pd.cut(dta['res'], bins = bins, labels = labels)
  dta['res'] = dta['res'].astype("int64")
  
  if binned:
    for col in list(dta)[:-1]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
