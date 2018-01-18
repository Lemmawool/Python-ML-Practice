# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:03:25 2018

@author: Trey
"""

import numpy as np

def featureNormalize(X):
  X_cols = len(X[0, ...])
  mu = np.zeros(X_cols).T
  X_norm = np.array(X, np.float64)
  sigma = np.array(mu)
  
  for i in range(X_cols):
    mu[i] = np.mean(X[..., i])
    sigma[i] = np.std(X[..., i], ddof=1)
    X_norm[..., i] = (1.0 * X[..., i] - mu[i]) / sigma[i]
  
  return X_norm



  