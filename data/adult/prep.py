# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:53:42 2018

@author: tobi_
"""

import pandas as pd

def prep_data(binned = False):

  colnames = ["Age",
  			"Workclass",
  			"Census_Weight",
  			"Education",
  			"Education_Numeric",
  			"Marital_Status",
  			"Occupation",
  			"Relationship",
  			"Race",
  			"Sex",
  			"Gain",
  			"Loss",
  			"Hours",
  			"Country",
  			"Income"]
  
  feat_cols = ["Age", 
           "Workclass", 
           "Education_Numeric", 
           "Marital_Status", 
           "Occupation", 
           "Relationship",
           "Race", 
           "Sex",
           "Gain",
           "Loss",
           "Hours",
           "Country",
           "Income"]
  
  
  urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"] 
  
  #dta_files = ["dta.csv","test.csv"]
  
  
  dta_train = pd.read_csv(urls[0],
                names = colnames,
                na_values=['?'],
                index_col = False,
                skipinitialspace = True)
  
  dta_test = pd.read_csv(urls[1],
                 names = colnames,
                 na_values=['?'],
                 index_col = False,
                 skipinitialspace = True)
  
  dta = dta_train.append(dta_test)
  dta = dta.loc[:,feat_cols]
  dta = dta.dropna()
  dta = dta.reset_index(drop = True)
  dta.Income = dta.Income.replace({"<=50K"  : 0, "<=50K."  : 0, ">50K." : 1, ">50K" : 1})
  dta["Age"] = pd.to_numeric(dta["Age"])
  dta = pd.get_dummies(dta)
  
  # dta.to_pickle("dta.pkl")

  if binned:
    bins = [0, 30, 40, 50, 65, 90]
    labels = [1,2,3,4,5]
    dta['Age'] = pd.cut(dta['Age'], bins = bins, labels = labels)
    dta['Age'] = dta['Age'].astype("int64")
    
    bins = [-1, 1, 1000, 2500, 5000, 10000, 30000, 100000]
    labels = [0,1,2,3,4,5,6]
    dta['Gain'] = pd.cut(dta['Gain'], bins = bins, labels = labels)
    dta['Gain'] = dta['Gain'].astype("int64")
    
    bins = [-1, 1, 1000, 2500, 5000, 10000, 30000, 100000]
    labels = [0,1,2,3,4,5,6]
    dta['Gain'] = pd.cut(dta['Gain'], bins = bins, labels = labels)
    dta['Gain'] = dta['Gain'].astype("int64")
    
    bins = [-1, 1, 1000, 2500, 5000]
    labels = [0,1,2,3]
    dta['Loss'] = pd.cut(dta['Loss'], bins = bins, labels = labels)
    dta['Loss'] = dta['Loss'].astype("int64")
    
    bins = [0, 20, 40, 60, 80, 100]
    labels = [1,2,3,4,5]
    dta['Hours'] = pd.cut(dta['Hours'], bins = bins, labels = labels)
    dta['Hours'] = dta['Hours'].astype("int64")
    
    bins = [0, 8, 13, 15, 16]
    labels = [1,2,3,4]
    dta['Education_Numeric'] = pd.cut(dta['Education_Numeric'], bins = bins, labels = labels)
    dta['Education_Numeric'] = dta['Education_Numeric'].astype("int64")
    
    
  return dta

  
# dta.to_pickle("dta_binned.pkl")

