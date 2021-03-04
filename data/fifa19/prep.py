# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "data.csv"

  dta = pd.read_csv(url,
              header = 0,
              skipinitialspace = True)
  
  dta = dta.drop(["Unnamed: 0", "ID", "Name", "Photo", "Flag", "Loaned From", "Club", "Club Logo", "Real Face", "Nationality", "Release Clause", "Value", "Jersey Number"], axis = 1)
  dta['Wage'] = dta['Wage'].apply(lambda x: float(str(x).lstrip('€').rstrip('K')))
  dta = dta.dropna()
  dta["Height"] = dta['Height'].apply(lambda x: (float(str(x).split('\'')[0])*12 + float(str(x).split('\'')[1]))*2.54)
  dta["Weight"] = dta["Weight"].apply(lambda t: int(t[:3])) 
  dta["Contract Valid Until"] = dta["Contract Valid Until"].apply(lambda t: int(t[-4:]) - 2018)
  dta["Joined"] = dta["Joined"].apply(lambda t: 2018 - int(t[-4:])) 
   
  
  pos_cols = ['CB',
               'RW',
               'LW',
               'CDM',
               'LM',
               'LF',
               'RCM',
               'CF',
               'CM',
               'LAM',
               'RS',
               'ST',
               'LB',
               'RDM',
               'RCB',
               'RAM',
               'LS',
               'RM',
               'LCM',
               'LWB',
               'RF',
               'CAM',
               'LCB',
               'RWB',
               'RB',
               'LDM']
  
  for col in pos_cols:
    dta[col] = dta[col].apply(lambda x: int(eval(x)))
    
  dta = pd.get_dummies(dta)
  
  bins = [0, 1, 3, 10, 1000]
  labels = [1,2,3,4]
  dta['Wage'] = pd.cut(dta['Wage'], bins = bins, labels = labels)
  dta['Wage'] = dta['Wage'].astype("int64")
  
  if binned:
    for col in list(dta)[:72]:
      if col == "Wage": 
        continue
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")