from ReLU import ReLU
import numpy as np
from mnist import MNIST

stride = 1

class Convolution():
    def __init__(self, height, width, size, R, k):
        self.height = height
        self.width = width
        self.size = size #batchsize
        self.R = R
        self.k = k
        self.padding = int(np.floor(R / 2))

    def convolute(self,w, x, b):
        return np.dot(w, x) + b

    def addpadding(self, x):
        pad = np.zeros((x.shape[0], self.padding))
        x = np.concatenate([pad, x], axis=1)
        pad = np.zeros((x.shape[0], self.padding))
        x = np.concatenate([x, pad], axis=1)
        pad = np.zeros((self.padding, x.shape[1]))
        x = np.concatenate([pad, x], axis=0)
        pad = np.zeros((self.padding, x.shape[1]))
        x = np.concatenate([x, pad], axis=0)
        return x

    def convertFilterOrder(self, x):
        fOrder = np.empty((self.R*self.R, self.width*self.height))
        for i in xrange(self.height):
            for j in xrange(self.width):
                fOrder[:, (i*self.height)+j] = x[i:i+self.R, j:j+self.R].ravel()
        return fOrder

    def XtoBatch(self, x):
        batch = np.empty((self.R*self.R, self.width * self.height * self.size))
        for i in xrange(x.shape[2]):
            y = self.createXmatrix(x[:,:,i])
            batch[:, i * (self.width * self.height) : (i+1) * (self.width * self.height)] = y
        return batch

    def createXmatrix(self, x):
        y1 = self.addpadding(x)
        y2 = self.convertFilterOrder(y1)
        return y2

    def BtoBatch(self, b):
        return b * np.array([1.0] * self.width * self.height * self.size)

    def convolution(self, x, w, b):
        x = self.XtoBatch(x)
        b = self.BtoBatch(b)
        y = self.convolute(w, x, b)
        return (x, y)

    def backOfConvolution(self, x, w, deltaY):
        deltaX = np.dot(w.T, deltaY)
        deltaW = np.dot(deltaY, x.T)
        deltaB = (np.sum(deltaY, axis=1)).reshape(-1,1)
        return (deltaX, deltaW, deltaB)

if __name__ == '__main__':
    PICT_HEIGHT = 28
    PICT_WIDTH = 28
    CLASS_SIZE = 10
    M_SIZE = 50
    B_SIZE = 100
    ETA = 0.01
    np.random.seed(200)

    def softmax(a):
        alpha = np.max(a, axis=0)
        return np.exp(a - alpha) / np.sum(np.exp(a - alpha), axis=0)

    def backOfSoftAndCross(ansY, y):
        return ((y - ansY) / B_SIZE)

    def crossEntropy(AnsY, y):
        return np.sum(AnsY * np.log(y) * -1, axis=0)

    mndata = MNIST("./le4nn/")
    X, Y = mndata.load_training()
    X = np.array(X)
    N = X.shape[0]
    X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
    Y = np.array(Y)
    """
    p, q = mndata.load_testing()
    p = np.array(p)
    q = np.array(q)
    p = p.reshape((p.shape[0], PICT_HEIGHT, PICT_WIDTH))
    """

    R = 3
    k1 = 32
    d = 2
    w1 = np.random.normal(0.0, 1.0 / (R * R), (k1, R * R))
    b1 = np.random.normal(0.0, 1.0 / (R * R), (k1, 1))

    w2 = np.random.normal(0.0, 1.0 / (M_SIZE), (M_SIZE, PICT_HEIGHT*PICT_WIDTH*k1))#
    b2 = np.random.normal(0.0, 1.0 / (M_SIZE), (M_SIZE, 1))

    w3 = np.random.normal(0.0, 1.0 / (M_SIZE), (CLASS_SIZE, M_SIZE))
    b3 = np.random.normal(0.0, 1.0 / (M_SIZE), (CLASS_SIZE, 1))

    relu = ReLU.ReLU()
    c1 = co.Convolution(PICT_HEIGHT, PICT_WIDTH, B_SIZE, R, k1)
    p1 = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, B_SIZE, k1)
    """
    if(os.path.exists("convolution.npz")):
        load_array = np.load("convolution.npz")
        w1 = load_array["w1"]
        b1 = load_array["b1"]
        w2 = load_array["w2"]
        b2 = load_array["b2"]
        w3 = load_array["w3"]
        b3 = load_array["b3"]
        load_array.close()
        """
    precision = 0
    for count in xrange(N / B_SIZE * 10 + 1):
        choice = np.random.choice(X.shape[0], B_SIZE)
        batchset = np.empty((X.shape[1] , X.shape[2], B_SIZE))
        ansY = np.empty((CLASS_SIZE, B_SIZE))
        answer = np.array([Y[i] for i in choice])
        j = 0
        for i in choice:
            batchset[:,:,j] = X[i]
            ansY[:,j] = np.array([0] * answer[j] + [1] + [0] * (10 - answer[j] - 1))
            j += 1
        x1, y1 = c1.convolution(batchset, w1, b1)
        z1 = relu.relu(y1)

        x2 = np.empty((k1 * PICT_HEIGHT * PICT_WIDTH, B_SIZE))
        t = np.split(z1, B_SIZE, axis=1)
        for i in xrange(B_SIZE):
            x2[:,i] = t[i].ravel()

        y2 = relu.fullyConnecterLayer(x2, w2, b2 * (np.array([1] * B_SIZE)))
        x3 = relu.relu(y2)
        y4 = relu.fullyConnecterLayer(x3, w3, b3 * (np.array([1] * B_SIZE)))
        res = softmax(y4)
        averageOfEntropy = np.sum(crossEntropy(ansY, res)) / B_SIZE
        recog = np.argmax(res, axis=0)
        precision += len(np.where(answer - recog == 0)[0]) * 1.0 / N
        if (count % (N / B_SIZE)) == 0:
            print count / (N / B_SIZE)
            print averageOfEntropy
            print precision
            precision = 0

        deltaSoft = backOfSoftAndCross(ansY, res)
        deltaX3, deltaW3, deltaB3 = backOfConnect(x3, w3, deltaSoft)
        deltaRelu2 = relu.backOfReLU(x3, deltaX3)
        deltaX2, deltaW2, deltaB2 = backOfConnect(x2, w2, deltaRelu2)
        deltareshape = np.empty((k1, deltaX2.shape[0]*deltaX2.shape[1]/k1))
        t = np.split(deltaX2, B_SIZE, axis=1)
        for i in xrange(B_SIZE):
            deltareshape[:,i * PICT_WIDTH * PICT_HEIGHT : (i+1) * PICT_WIDTH * PICT_HEIGHT] = t[i].reshape(k1, -1)
        deltaRelu1 = relu.backOfReLU(z1, deltareshape)
        deltaX1, deltaW1, deltaB1 = c1.backOfConvolution(x1, w1, deltaRelu1)

        w1 -= (deltaW1 * ETA)
        b1 -= (deltaB1 * ETA)
        w2 -= (deltaW2 * ETA)
        b2 -= (deltaB2 * ETA)
        w3 -= (deltaW3 * ETA)
        b3 -= (deltaB3 * ETA)


#    np.savez('convolution32-3.npz', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
