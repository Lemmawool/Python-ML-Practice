# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:17:22 2018

@author: Trey
"""

import numpy as np

def normalEquation(X, y):
  X_t = X.T
  return np.linalg.pinv(X_t @ X) @ X_t @ y