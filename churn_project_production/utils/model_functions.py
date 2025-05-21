#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 20:24:01 2025

@author: carmenarnau
"""

from scipy.stats import chi2_contingency
from sklearn.metrics import auc, roc_curve
import pandas as pd
import numpy as np

def chi_square(df, target, input_cols, threshold):

    statistical_significance=[]

    for attr in input_cols:
        data_count=pd.crosstab(df[attr],df[target])
        obs=data_count.values
   
        chi2, p, dof, expected = chi2_contingency(obs)

        statistical_significance.append([attr,np.round(p,4)])

    statistical_significance=pd.DataFrame(statistical_significance)
    statistical_significance.columns=["Attribute","P-value"]

    df_mod_cols = statistical_significance[statistical_significance["P-value"]<threshold].Attribute.tolist() 
  
    return df_mod_cols, statistical_significance




def predict_and_get_auc(model, X_train, X_test, y_train, y_test):
    
    y_train_prob = model.predict_proba(X_train)
    y_test_prob = model.predict_proba(X_test)

    fpr, tpr, threshold = roc_curve(y_train, y_train_prob[:, 1])
    print("AUC train = ", round(auc(fpr, tpr), 2))

    fpr, tpr, threshold = roc_curve(y_test, y_test_prob[:, 1])
    print("AUC test = ", round(auc(fpr, tpr), 2))

