# coding: UTF-8
# 2次元のパーセプトロンの学習規則の実装例
import numpy as np
import matplotlib.pyplot as plt
import sys

############################################################
#                                                          #
#          #####         #      #      #      ##           #
#          #    #       ##      ##     #     #  #          #
#          #    #      # #      # #    #    #    #         #
#          #####      #  #      #  #   #    #    #         #
#          #         #####      #   #  #    #    #         #
#          #        #    #      #    # #    #    #         #
#          #       #     #      #     ##     #  #          #
#          #      #      #      #      #      ##           #
#                                                          #
#                                      ### ###  ## # #     #
#                                       #  #++ #   ###     #
#                                       #  ###  ## # #     #
#                                                          #
############################################################


def train(wvec, xvec, label):
    low = 0.5#学習係数
    if (np.dot(wvec,xvec) * label < 0):
        wvec_new = wvec + label*low*xvec
        return wvec_new
    else:
        return wvec

if __name__ == '__main__':

    train_num = 100#学習データ数

    #class1の学習データ
    x1_1=np.random.rand(int(train_num/2)) * 5 + 1 #x成分
    x1_2=np.random.rand(int(train_num/2)) * 5 + 1 #y成分
    label_x1 = np.ones(int(train_num/2)) #ラベル（すべて1）

    #class2の学習データ
    x2_1=(np.random.rand(int(train_num/2)) * 5 + 1) * -1 #x成分
    x2_2=(np.random.rand(int(train_num/2)) * 5 + 1) * -1 #y成分
    label_x2 = np.ones(int(train_num/2)) * -1 #ラベル（すべて-1）

    x0=np.ones(int(train_num/2)) # x0は常に1
    x1=np.c_[x0, x1_1, x1_2]
    x2=np.c_[x0, x2_1, x2_2]

    xvecs=np.r_[x1, x2]
    labels = np.r_[label_x1, label_x2]

    #wvec = np.array([2,-1,3])#初期の重みベクトル 適当に決める
    wvec = np.array([-16,-1,-1])#初期の重みベクトル 適当に決める

    loop = 100
    for j in range(loop):
        for xvec, label in zip(xvecs, labels):
            wvec = train(wvec, xvec, label)

    print(wvec)

    plt.scatter(x1[:,1], x1[:,2], c='red', marker="o")
    plt.scatter(x2[:,1], x2[:,2], c='yellow', marker="x")
    #分離境界線
    x_fig = np.array(range(-8,8))
    y_fig = -(wvec[1]/wvec[2])*x_fig - (wvec[0]/wvec[2])

    plt.plot(x_fig,y_fig)
    plt.show()