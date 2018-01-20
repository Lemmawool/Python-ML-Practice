# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:56:32 2018

@author: Trey
"""

import math
import numpy as np
import scipy as sp

def sigmoid(z):
  return 1 / (1 + (math.e ** (-z)))

def costFunc(X, y, theta):
  m = len(y)
  J = (1/m) * (-y.T @ sp.log(sigmoid(X @ theta)) - (1 - y).T @ sp.log(1 - sigmoid(X @ theta)))
  return J

def gradFunc(X, y, theta):
  m = len(y)
  grad = (1/m) * X.T @ (sigmoid (X @ theta) - y)
  return grad

def costFunc2(theta, X, y):
  theta = np.array([theta]).T
  m = len(y)
  J = (1/m) * (-y.T @ sp.log(sigmoid(X @ theta)) - (1 - y).T @ sp.log(1 - sigmoid(X @ theta)))
  return J[0][0]

def gradFunc2(theta, X, y):
  m = len(y)
  grad = (1/m) * X.T @ (sigmoid (X @ theta) - y)
  return np.ndarray.flatten(grad)



