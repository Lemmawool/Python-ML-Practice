# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:51:28 2018

@author: Trey

Practicing machine learning with python running gradient descent with one feature.

"""

import numpy as np
import computeCost as cc
import gradientDescent as gd

data = np.loadtxt(open("data1.txt", "rb"), delimiter=",")
X = data[..., 0]
y = np.array([data[..., 1]]).T
m = len(y)
X = np.array([np.ones(m), X]).T

theta = np.zeros((2,1))
J = cc.computeCost(X, y, theta)
print("With theta = [0 0] the cost coumputed is: " + str(J))

theta = np.zeros((2,1))
theta = gd.gradientDescent(X, y, theta, 0.01, 2000)
print("Value of theta: " + str(theta))


