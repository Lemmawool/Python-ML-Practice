# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:01:30 2018

@author: Trey
"""

import numpy as np
import ex2_functions as ex2

data = np.loadtxt(open("data1.txt", "rb"), delimiter=",")
num_features = len(data[1, ...])
X = data[..., range(num_features - 1)]
y = np.array([data[..., num_features - 1]]).T
m = len(y)
X = np.append(np.ones((m,1)), X, axis=1)
theta = np.zeros((num_features,1))

cost = ex2.costFunc(X, y, theta)
print(cost)
grad = ex2.gradFunc(X, y, theta)
print(grad)




