# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:51:28 2018

@author: Trey

Practicing machine learning with python running gradient descent with one feature.

"""

import numpy as np
import scipy as sp
import computeCost as cc
import gradientDescent as gd

data = np.loadtxt(open("data1.txt", "rb"), delimiter=",")
X = data[..., 0]
y = np.transpose(np.asmatrix(data[..., 1]))
m = len(y)
X = np.transpose(np.array([np.ones(m), X]))
theta = np.matlib.zeros((2,1))

J = cc.computeCost(X, y, theta)
print("With theta = [0 0] the cost coumputed is: " + str(J))
theta = gd.gradientDescent(X, y, theta, 0.01, 2000)
print(theta)



