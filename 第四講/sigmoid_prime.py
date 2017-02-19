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

# 日本語を使うため必要
fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")

# 新規のウィンドウを描画
fig = plt.figure()

# サブプロットを追加
#ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(1,2,1)
ax3 = fig.add_subplot(1,2,2)

x = np.arange(-3, 3, 0.1)
y = np.sin(x)
#ax1.plot(x, y)

y1 = sigmoid(x)
ax2.set_ylim(0,1)
ax2.plot(x, y1)

y2 = sigmoid_prime(x)
ax3.set_ylim(0,1)
ax3.plot(x, y2)

plt.show()
plt.savefig("image.png")