# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "data.csv"

  dta = pd.read_csv(url,
              header = 0,
              skipinitialspace = True)
  
  dta = dta.dropna()
  dta = dta.drop(["Name","Publisher","Global_Sales","Developer"], axis = 1)
  dta = pd.get_dummies(dta)
  
  bins = [0, 72, 100]
  labels = [0,1]
  dta['Critic_Score'] = pd.cut(dta['Critic_Score'], bins = bins, labels = labels)
  dta['Critic_Score'] = dta['Critic_Score'].astype("int64")
  
  if binned:
    for col in ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Count', 'User_Count']:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
