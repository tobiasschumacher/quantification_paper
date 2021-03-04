# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:41:07 2018

@author: tobi_
"""


import pandas as pd

def prep_data(binned = False):

  dta_white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
              index_col = False,
              sep = ';',
              skipinitialspace = True)
  dta_white.insert(0, 'type', 'white')
  
  dta_red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
              index_col = False,
              sep = ';',
              skipinitialspace = True)
  dta_red.insert(0, 'type', 'red')

  dta = dta_white.append(dta_red)  
  
  dta = pd.get_dummies(dta)
  
  bins = [0, 5, 10]
  labels = [0,1]
  dta['quality'] = pd.cut(dta['quality'], bins = bins, labels = labels)
  dta['quality'] = dta['quality'].astype("int64")
  
  #dta.to_pickle("dta.pkl")
   
  if binned:
    bins = [0, 10, 11, 12, 15]
    labels = [1,2,3,4]
    dta['alcohol'] = pd.cut(dta['alcohol'], bins = bins, labels = labels)
    dta['alcohol'] = dta['alcohol'].astype("int64")
    
    bins = [0, 6, 8, 16]
    labels = [1,2,3]
    dta['fixed acidity'] = pd.cut(dta['fixed acidity'], bins = bins, labels = labels)
    dta['fixed acidity'] = dta['fixed acidity'].astype("int64")
    
    bins = [0, 0.2, 0.3, 0.5, 2]
    labels = [1,2,3,4]
    dta['volatile acidity'] = pd.cut(dta['volatile acidity'], bins = bins, labels = labels)
    dta['volatile acidity'] = dta['volatile acidity'].astype("int64")
    
    bins = [-0.1, 0.2, 0.3, 0.5, 2]
    labels = [1,2,3,4]
    dta['citric acid'] = pd.cut(dta['citric acid'], bins = bins, labels = labels)
    dta['citric acid'] = dta['citric acid'].astype("int64")
    
    bins = [0, 2, 3, 5, 10,100]
    labels = [1,2,3,4,5]
    dta['residual sugar'] = pd.cut(dta['residual sugar'], bins = bins, labels = labels)
    dta['residual sugar'] = dta['residual sugar'].astype("int64")
    
    bins = [0, 0.03, 0.05, 0.1, 1]
    labels = [1,2,3,4]
    dta['chlorides'] = pd.cut(dta['chlorides'], bins = bins, labels = labels)
    dta['chlorides'] = dta['chlorides'].astype("int64")
    
    bins = [0, 20, 30, 40, 50, 300]
    labels = [1,2,3,4,5]
    dta['free sulfur dioxide'] = pd.cut(dta['free sulfur dioxide'], bins = bins, labels = labels)
    dta['free sulfur dioxide'] = dta['free sulfur dioxide'].astype("int64")
    
    bins = [0, 100, 200, 500]
    labels = [1,2,3]
    dta['total sulfur dioxide'] = pd.cut(dta['total sulfur dioxide'], bins = bins, labels = labels)
    dta['total sulfur dioxide'] = dta['total sulfur dioxide'].astype("int64")
    
    bins = [0, 0.99, 1, 2]
    labels = [1,2,3]
    dta['density'] = pd.cut(dta['density'], bins = bins, labels = labels)
    dta['density'] = dta['density'].astype("int64")
    
    bins = [0, 3, 3.5, 5]
    labels = [1,2,3]
    dta['pH'] = pd.cut(dta['pH'], bins = bins, labels = labels)
    dta['pH'] = dta['pH'].astype("int64")
    
    bins = [0, 0.4, 0.5, 0.6, 3]
    labels = [1,2,3,4]
    dta['sulphates'] = pd.cut(dta['sulphates'], bins = bins, labels = labels)
    dta['sulphates'] = dta['sulphates'].astype("int64")
    
    # dta.to_pickle("dta_binned.pkl")
    
  return dta