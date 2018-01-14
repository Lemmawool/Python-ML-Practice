# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:51:28 2018

@author: Trey

Practicing machine learning with python running gradient descent with one feature.

"""

import numpy as np
import scipy as sp
import computeCost 

data = np.loadtxt(open("data1.txt", "rb"), delimiter=",")
X = data[..., 0]
y = np.transpose(np.asmatrix(data[..., 1]))
m = len(y)

X = np.transpose(np.array([np.ones(m), X]))
theta = np.matlib.zeros((2,1))

J = computeCost.computeCost(X, y, theta)
print(J)





