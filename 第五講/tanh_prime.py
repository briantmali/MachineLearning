import numpy as np
from pandas import *
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numpy.random import randn

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
    
def tanh(x):
    return 2*sigmoid(2*x) -1

def tanh_prime(x):
    f = tanh(x)
    return 1-f*f

# 日本語を使うため必要
fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")

# 新規のウィンドウを描画
fig = plt.figure()

# サブプロットを追加
ax1 = fig.add_subplot(2,2,1)
ax4 = fig.add_subplot(2,2,2)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,4)

x = np.arange(-3, 3, 0.1)
y = sigmoid(x)
ax1.set_ylim(-1,1)
ax1.set_title('sigmoid')
ax1.plot(x, y)
ax1.axhline(y=.0, xmin=-1, xmax=1)
ax1.axvline(x=.0, ymin=-3, ymax=3)

y4 = sigmoid_prime(x)
ax4.set_ylim(-1,1)
ax4.set_title('sigmoid_prime')
ax4.plot(x, y4)
ax4.axhline(y=.0, xmin=-1, xmax=1)
ax4.axvline(x=.0, ymin=-3, ymax=3)


y1 = tanh(x)
ax2.set_ylim(-1,1)
ax2.set_title('tanh')
ax2.plot(x, y1)
ax2.axhline(y=.0, xmin=-1, xmax=1)
ax2.axvline(x=.0, ymin=-3, ymax=3)

y2 = tanh_prime(x)
ax3.set_ylim(-1,1)
ax3.set_title('tanh_prime')
ax3.plot(x, y2)
ax3.axhline(y=.0, xmin=-1, xmax=1)
ax3.axvline(x=.0, ymin=-3, ymax=3)

plt.show()
plt.savefig("image.png")