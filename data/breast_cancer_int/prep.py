# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 21:52:35 2018

@author: tobi_
"""

import pandas as pd

def prep_data():
  
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
  
  colnames = ["ID",
          "Clump_Thickness",
          "Uniformity_of_Cell_Size",
          "Uniformity_of_Cell_Shape",
          "Marginal_Adhesion",
          "Single_Epithelial_Cell_Size",
          "Bare_Nuclei",
          "Bland_Chromatin",
          "Normal_Nucleoli",
          "Mitoses",
          "Class"]
  
  
  dta = pd.read_csv(url, 
              names = colnames, 
              na_values=['?'], 
              index_col = "ID", 
              skipinitialspace = True)
  
  dta = dta.dropna()
  dta.Class = dta.Class.replace({2  : 0, 4 : 1})  
  return dta
  
# dta.to_pickle("dta.pkl")
