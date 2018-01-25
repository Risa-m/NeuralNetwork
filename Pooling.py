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
#        self.choice = np.empty((k, self.height*self.width*self.size))
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

    def pooling(self, x):
#        print x # k , height*width*size
        x2 = np.array(np.split(x, self.size, axis=1))
#        print x2 # size, k, height*width
        x3 = x2.reshape(self.size, self.k, self.height, self.width)
#        print x3 # size, k, height, width
        x4 = np.array(np.split(x3, self.width/self.d, axis=3))
#        print x4 # width/d, size, k, height, d(width)
        x5 = np.array(np.split(x4, self.height/self.d, axis=3))
#        print x5 # height/d, width/d, size, k, d(height) ,d(height)
        x52 = x5.reshape(self.height/self.d, self.width/self.d, self.size, self.k, self.d*self.d)
#        print x52

        x522 = np.max(x52, axis=4)
        expand = np.ones((self.height/self.d, self.width/self.d, self.size, self.k, self.d, self.d))
        x622 = x522.reshape(self.height/self.d, self.width/self.d, self.size, self.k, 1, 1)
        x722 = expand * x622

        x52 = x52.reshape(self.height/self.d, self.width/self.d, self.size, self.k, self.d, self.d)
#        print x522
        x51 = np.zeros((self.height/self.d, self.width/self.d, self.size, self.k, self.d*self.d))
        x53 = np.where(x52-x722==0, 1, 0)
#        x51[:,:,:,:,np.argmax(x52, axis=4)] = 1.0
#        x52[x52!=1] = 0.0t
#        print x52
        x6 = np.concatenate([x53[i] for i in xrange(self.height/self.d)], axis=3)
#        print x6 #

        x7 = np.concatenate([x6[i] for i in xrange(self.width/self.d)], axis=3)
#        print x7 #size, k, height, width
        x8 = x7.reshape(self.size, self.k, self.height*self.width)
        x9 = np.concatenate([x8[i] for i in xrange(self.size)], axis=1)
#        print x9 # k , height*width*size

        x62 = x522.reshape(self.height/self.d, self.width/self.d, self.size, self.k, 1, 1)
#        print x62
        x72 = np.concatenate([x62[i] for i in xrange(self.height/self.d)], axis=3)
#        print x72.shape
        x82 = np.concatenate([x72[i] for i in xrange(self.width/self.d)], axis=3)
#        print x82.shape
#        x92 = x82.reshape(self.size, self.k, self.height/self.d*self.width/self.d)

#        x102 = np.concatenate([x92[i] for i in xrange(self.size)], axis=1)
        x92 = x82.reshape(self.size, self.k* self.height/self.d*self.width/self.d)
#        x102 = x92.T

        return (x92.T, x9)

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


    def backOfPooling(self, x, choice):
        expand = np.ones((self.height/self.d, self.width/self.d, self.size, self.k, self.d, self.d))
        x2 = x.T
#        print x2
#        print x2.shape # size, height*width*k/d/d
        x3 = x2.reshape((self.size, self.k, self.height/self.d, self.width/self.d))
#        print x3
#        print x3.shape
        x4 = np.array(np.split(x3, self.width/self.d, axis=3))
#        print x4
#        print x4.shape # width/d, size, k, height/d, 1
        x5 = np.array(np.split(x4, self.height/self.d, axis=3))
        x6 = x5.reshape(self.height/self.d, self.width/self.d, self.size, self.k, 1, 1)
#        print x6

        x7 = expand * x6
#        print expand
        x8 = np.concatenate([x7[i] for i in xrange(self.height/self.d)], axis=3)
        x9 = np.concatenate([x8[i] for i in xrange(self.width/self.d)], axis=3)
        x10 = x9.reshape(self.size, self.k, self.height*self.width)
#        print x10.shape
        x11 = np.concatenate([x10[i] for i in xrange(self.size)], axis=1)
#        print x11.shape #k , size*height*width
        back = x11
#        print back * choice
        return back * choice



    def batchtodimention(self, x):
        y = np.empty((self.k * (self.height / self.d) * (self.width / self.d), self.size))
        t = np.split(x.copy(), self.size, axis=1)
        for i in xrange(self.size):
            y[:,i] = t[i].ravel()
        return y

        """
    def batchtodimention(self, x):
        y = np.empty((x.shape[0], self.height / self.d, self.width / self.d, self.size))
        t = np.split(x, self.size, axis=1)
        for i in xrange(self.size):
            y[:,:,:,i] = t[i].reshape((x.shape[0], self.height / self.d, self.width / self.d))
        return x
        """


if __name__ == '__main__':
    PICT_HEIGHT = 28
    PICT_WIDTH = 28

    mndata = MNIST("./le4nn/")
    X, Y = mndata.load_training()
    X = np.array(X)
    X_SIZE = X.shape[1]
    N = X.shape[0]
    X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
    Y = np.array(Y)
    """
    p, q = mndata.load_testing()
    p = np.array(p)
    q = np.array(q)
    p = p.reshape((p.shape[0], PICT_HEIGHT, PICT_WIDTH))
    """
    l = le.Learning()
    leru = ReLU.ReLU()
#    c1 = co.Convolution(28, 28, B_SIZE)
#    c2 = co.Convolution(14,14)

#    y1 = c1.convolutetest(X)
#    y2 = leru.relu(y1)
    d = 2
    height = 8
    width = 6
    B_SIZE = 2
    k = 3
    p = Pooling(d, height, width, B_SIZE, k)

    arr = np.arange(height*width*B_SIZE*k).reshape(k,-1)
    z2, choice = p.pooling(arr)
    z3 = p.backOfPooling(z2, choice)
#    z3 = p.batchtodimention(z2)
#    y2 = c2.convolutetest(z1)
#    z2 = p.pooling(y2, co.k)
"""
    import matplotlib.pyplot as plt
    from pylab import cm
    t = z3.reshape((1, PICT_HEIGHT / 2, PICT_WIDTH / 2, 1))
#    t = np.hstack((t[0,:,:,3], t[1,:,:,3], t[2,:,:,3], t[3,:,:,3]))
    plt.imshow(t[1,:,:,1], cmap=cm.gray)
    plt.show()
"""
