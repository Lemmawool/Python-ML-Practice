# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:44:26 2018

@author: Trey
"""

import numpy as np

def computeCost(X, y, theta):
  m = len(y)
  return ((1/(2*m)) * sum(np.square(X @ theta - y)))