# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:02:17 2018

@author: Trey
"""

import numpy as np
import gradientDescent as gd
import featureNormalize as fn

data = np.loadtxt(open("data2.txt", "rb"), delimiter=",")

num_features = len(data[1, ...])
X = data[..., range(num_features - 1)]
y = np.array([data[..., num_features - 1]]).T
m = len(y)
X_norm = fn.featureNormalize(X)
X_norm = np.append(np.ones((m,1)), X_norm, axis=1)

theta = np.zeros((3,1))
theta = gd.gradientDescent(X_norm, y, theta, 0.01, 400)
print(theta)
test = np.array([[1650,3]]).T
test = fn.featureNormalize(test)
test = np.insert(test, 0, [[1]])
print(test @ theta)




