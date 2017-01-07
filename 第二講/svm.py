import numpy
from matplotlib import pyplot
import sys

def f(x, y):
    return x - y

if __name__ == '__main__':

    param = sys.argv

    numpy.random.seed()
    N = 30
    d = 2
    X = numpy.random.randn(N, d)
    T = numpy.array([1 if f(x, y) > 0 else - 1 for x, y in X])
    alpha = numpy.zeros(N)
    beta = 1.0
    eta_al = 0.0001 # update ratio of alpha
    eta_be = 0.1 # update ratio of beta
    itr = 1000

    for _itr in range(itr):
        for i in range(N):
            delta = 1 - (T[i] * X[i]).dot(alpha * T * X.T).sum() - beta * T[i] * alpha.dot(T)
            alpha[i] += eta_al * delta
        for i in range(N):
            beta += eta_be * alpha.dot(T) ** 2 / 2

    index = alpha > 0
    w = (alpha * T).T.dot(X)
    b = (T[index] - X[index].dot(w)).mean()

    if '-d' in param or '-s' in param:
        seq = numpy.arange(-3, 3, 0.02)
        pyplot.figure(figsize = (6, 6))
        pyplot.xlim(-3, 3)
        pyplot.ylim(-3, 3)
        pyplot.plot(seq, -(w[0] * seq + b) / w[1], 'k-')
        pyplot.plot(X[T ==  1,0], X[T ==  1,1], 'ro')
        pyplot.plot(X[T == -1,0], X[T == -1,1], 'bo')

        if '-s' in param:
            pyplot.savefig('graph.png')

        if '-d' in param:
            pyplot.show()