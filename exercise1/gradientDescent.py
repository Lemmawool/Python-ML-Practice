# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:47:52 2018

@author: Trey
"""

import numpy as np

def gradientDescent(X, y, theta, alpha, interations):
  m = len(y)
  n = len(X[0, ...])
  temp = np.array(X[0, ...])
  
  for i in range(interations):
    for j in range(n):
      temp[j] = theta[j] - alpha * (1/m) * sum(np.transpose(X * theta - y) * np.transpose(np.asmatrix(X[..., j])))
    theta = np.transpose(np.matrix(np.array(temp)))
  
  return theta