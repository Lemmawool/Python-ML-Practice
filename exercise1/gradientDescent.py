# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:47:52 2018

@author: Trey
"""

import numpy as np

def gradientDescent(X, y, theta, alpha, interations):
  m = len(y)
  temp = X[0, ...]
  
  for i in range(interations):
    for j in range(len(X[0, ...])):
      temp[j] = theta[j] - alpha * (1/m) * sum(np.transpose(X * theta - y) * np.transpose(np.asmatrix(X[..., j])))
      #print((np.transpose(np.asmatrix(X[..., j]))).sum())
    theta = temp
  
  return theta