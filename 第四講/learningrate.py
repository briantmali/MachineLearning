# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]
def g(x):
    return np.array([2 * x[0], 100 * x[1]])

xi = np.linspace(-200,200,1000)
yi = np.linspace(-100,100,1000)

X,Y = np.meshgrid(xi, yi)
Z = X * X + 50 * Y * Y

def contour(X,Y,Z, arr = None):
    plt.figure(figsize=(15,7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0,0,marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i+2,0],arr[i:i+2,1])
        
#contour(X,Y,Z)
#plt.show()

def gd(x_start, step, g):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step
        
        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

#case 1
#res, x_arr = gd([150,75], 0.016, g)
#contour(X,Y,Z, x_arr)

#case 2
res, x_arr = gd([150,75], 0.019, g)
#res, x_arr = gd([150,75], 0.020, g)
#res, x_arr = gd([150,75], 0.021, g)
contour(X,Y,Z, x_arr)


plt.show()