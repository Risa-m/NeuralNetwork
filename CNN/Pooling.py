import Learning as le
import ReLU
import Convolution as co
import numpy as np
from mnist import MNIST
import time


class Pooling():
    def __init__(self, d, height, width, B_SIZE, k):
        self.d = d
        self.height = height
        self.width = width
        self.size = B_SIZE
        self.k = k
    def pooling(self, x):
        x2 = np.array(np.split(x, self.size, axis=1))
        x3 = x2.reshape(self.size, self.k, self.height, self.width)
        x4 = np.array(np.split(x3, self.width/self.d, axis=3))
        x5 = np.array(np.split(x4, self.height/self.d, axis=3))
        x52 = x5.reshape(self.height/self.d, self.width/self.d, self.size, self.k, self.d*self.d)

        x522 = np.max(x52, axis=4)
        expand = np.ones((self.height/self.d, self.width/self.d, self.size, self.k, self.d, self.d))
        x622 = x522.reshape(self.height/self.d, self.width/self.d, self.size, self.k, 1, 1)
        x722 = expand * x622

        x52 = x52.reshape(self.height/self.d, self.width/self.d, self.size, self.k, self.d, self.d)
        x51 = np.zeros((self.height/self.d, self.width/self.d, self.size, self.k, self.d*self.d))
        x53 = np.where(x52-x722==0, 1, 0)
        x6 = np.concatenate([x53[i] for i in xrange(self.height/self.d)], axis=3)

        x7 = np.concatenate([x6[i] for i in xrange(self.width/self.d)], axis=3)
        x8 = x7.reshape(self.size, self.k, self.height*self.width)
        x9 = np.concatenate([x8[i] for i in xrange(self.size)], axis=1)

        x62 = x522.reshape(self.height/self.d, self.width/self.d, self.size, self.k, 1, 1)
        x72 = np.concatenate([x62[i] for i in xrange(self.height/self.d)], axis=3)
        x82 = np.concatenate([x72[i] for i in xrange(self.width/self.d)], axis=3)
        x92 = x82.reshape(self.size, self.k* self.height/self.d*self.width/self.d)
        return (x92.T, x9)

    def backOfPooling(self, x, choice):
        expand = np.ones((self.height/self.d, self.width/self.d, self.size, self.k, self.d, self.d))
        x2 = x.T
        x3 = x2.reshape((self.size, self.k, self.height/self.d, self.width/self.d))
        x4 = np.array(np.split(x3, self.width/self.d, axis=3))
        x5 = np.array(np.split(x4, self.height/self.d, axis=3))
        x6 = x5.reshape(self.height/self.d, self.width/self.d, self.size, self.k, 1, 1)
        x7 = expand * x6
        x8 = np.concatenate([x7[i] for i in xrange(self.height/self.d)], axis=3)
        x9 = np.concatenate([x8[i] for i in xrange(self.width/self.d)], axis=3)
        x10 = x9.reshape(self.size, self.k, self.height*self.width)
        x11 = np.concatenate([x10[i] for i in xrange(self.size)], axis=1)
        back = x11
        return back * choice


        ##### very slow
    def pooling2(self, x):
        pool = np.empty((self.k, (self.height/self.d)*(self.width/self.d) * self.size))
        choice = np.zeros((x.shape[0], x.shape[1]))
        for l in xrange(self.k):
            for m in xrange(self.size):
                for i in xrange(self.height/self.d):
                    for j in xrange(self.width/self.d):
                        pool = self.poolingOfList(x, pool, l, m, i, j)
                        index = self.poolingOfChoice(x, l, m, i, j)
                        choice[l, index] = 1.0
        return (self.batchtodimention(pool), choice)

    def poolingOfList(self, x, pool, l, m, i, j):
        pool[l, m * (self.width / self.d) * (self.height / self.d) + i * (self.width / self.d) + j] = np.max(x[[l], [m * self.width * self.height + (i * self.d + a) * self.width + j * self.d + b for a in xrange(self.d) for b in xrange(self.d)]])
        return pool

    def poolingOfChoice(self, x, l, m, i, j):
        y = np.argmax(x[[l], [m * self.width * self.height + (i * self.d + a) * self.width + j * self.d + b for a in xrange(self.d) for b in xrange(self.d)]])
        for i2 in xrange(self.d):
            for j2 in xrange(self.d):
                if y == i2 * self.d + j2:
                    t = m * self.width * self.height + (i * self.d + i2) * self.width + j * self.d + j2
        return t

    def backOfPooling2(self, x, choice):
        back = np.empty((choice.shape[0], choice.shape[1]))
        for l in xrange(self.k):
            for m in xrange(self.size):
                for i in xrange(self.height/self.d):
                    for j in xrange(self.width/self.d):
                        back[[l], [m * (self.width) * (self.height) + (i * self.d + a) * (self.width) + j * self.d + b for a in xrange(self.d) for b in xrange(self.d)]] = x[l * (self.height / self.d) * (self.width / self.d) + i * (self.height / self.d) + j, m]
        return back * choice

    def batchtodimention(self, x):
        y = np.empty((self.k * (self.height / self.d) * (self.width / self.d), self.size))
        t = np.split(x.copy(), self.size, axis=1)
        for i in xrange(self.size):
            y[:,i] = t[i].ravel()
        return y
