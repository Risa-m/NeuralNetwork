import os.path
from Graph import Graph
import numpy as np
from mnist import MNIST
import time


PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
########## ##########
class Learning(object):
    Epoch_Size = 21
    mndata = MNIST("./le4nn/")
    X, Y = mndata.load_training()
    X = np.array(X)
    Y = np.array(Y)

    p, q = mndata.load_testing()
    p = np.array(p)
    q = np.array(q)

    X_SIZE = X.shape[1]
    N = X.shape[0]

    def __init__(self):
        np.random.seed(200)
        self.w1 = np.random.normal(0.0, 1.0 / Learning.X_SIZE, (M_SIZE, Learning.X_SIZE))
        self.b1 = np.random.normal(0.0, 1.0 / Learning.X_SIZE, (M_SIZE, 1))
        self.w2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE, M_SIZE))
        self.b2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE,1))

    # x is 10000 * 784 vector
    def inputLayer(self, x):
        return x

    def fullyConnecterLayer(self, x, w, b):
        return np.dot(w, x.reshape(-1, 1)) + b

    def sigmoid(self, x):
        sigmoid_range = 34.538776394910684
        t = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        alpha = max(a)
        return np.exp(a - alpha) / np.sum(np.exp(a - alpha))

    def recogRes(self, y):
        return np.argmax(y)

    def forward(self, x):
        layer1Out = self.inputLayer(x)
        layer2In = self.fullyConnecterLayer(layer1Out, self.w1, self.b1)
        layer2Out = self.sigmoid(layer2In)
        layer3In = self.fullyConnecterLayer(layer2Out, self.w2, self.b2)
        layer3Out = self.softmax(layer3In)
        return (layer1Out, layer2Out, layer3Out)

    ##########

    def crossEntropy(self, AnsY, y):
        return np.dot(AnsY, np.log(y)) * -1

    ##########

    def backOfSoftAndCross(self, ansY, y):
        return ((y.T - ansY) / B_SIZE)

    def backOfConnect(self, x, w, deltaY):
        deltaX = np.dot(w.T, deltaY)
        deltaW = np.dot(deltaY, x.T)
        deltaB = np.sum(deltaY, axis=1).reshape(-1, 1)
        return (deltaX, deltaW, deltaB)

    def backOfSig(self, x, deltaY):
        return (1 - x) * x * deltaY

    def backPropagate(self, inputX1, inputX2, deltaA):
        deltaX2, deltaW2, deltaB2 = self.backOfConnect(inputX2, self.w2, deltaA)
        deltaSig = self.backOfSig(inputX2, deltaX2)
        _, deltaW1, deltaB1 = self.backOfConnect(inputX1, self.w1, deltaSig)
        return (deltaW1, deltaB1, deltaW2, deltaB2)

    def renewParam(self, deltaW1, deltaB1, deltaW2, deltaB2):
        self.w1 -= (deltaW1 * ETA)
        self.b1 -= (deltaB1 * ETA)
        self.w2 -= (deltaW2 * ETA)
        self.b2 -= (deltaB2 * ETA)

    ########## ##########

    def importFile(self, filename):
        if(os.path.exists("%s.npz" % filename)):
            load_array = np.load("%s.npz" % filename)
            self.w1 = load_array["w1"]
            self.b1 = load_array["b1"]
            self.w2 = load_array["w2"]
            self.b2 = load_array["b2"]
            load_array.close()

    def saveFile(self, filename):
        np.savez('%s.npz' % filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def test(self):
        correct = 0
        for i in xrange(Learning.p.shape[0]):
            inputX = Learning.p[i] / 256.0
            x, y1, y2 = self.forward(inputX)
            correct = correct + (1.0 / Learning.p.shape[0]) if(self.recogRes(y2)==Learning.q[i]) else correct
        return correct



if __name__ == '__main__':
    l = Learning()
    count = 0
    precision = 0
    inputX1 = np.empty((l.X_SIZE, B_SIZE))
    inputX2 = np.empty((M_SIZE, B_SIZE))
    deltaA = np.empty((CLASS_SIZE, B_SIZE))
    for count in xrange(l.N / B_SIZE * 21):
        minibatch = np.random.choice(l.N, B_SIZE)
        averageOfEntropy = 0
        correct = 0
        j = 0
        for i in minibatch:
            inputX = l.X[i] / 256.0
            x, y1, y2 = l.forward(inputX)
            ansY = [0] * l.Y[i] + [1] + [0] * (10 - l.Y[i] - 1)
            averageOfEntropy += l.crossEntropy(ansY, y2) / B_SIZE
            inputX1[:, j] = x
            inputX2[:, j] = y1.ravel()
            deltaA[:, j] = l.backOfSoftAndCross(ansY, y2)
            correct = correct + (1.0 / B_SIZE) if(l.recogRes(y2) == l.Y[i]) else correct
            j += 1

        deltaW1, deltaB1, deltaW2, deltaB2 = l.backPropagate(inputX1, inputX2, deltaA)
        l.renewParam(deltaW1, deltaB1, deltaW2, deltaB2)
        precision += correct / (l.N / B_SIZE)

        if (count % (l.N / B_SIZE)) == 0:
            print count / (l.N / B_SIZE)
            print averageOfEntropy
            print precision
            precision = 0
        count += 1
