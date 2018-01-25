import Learning as le
from Graph import Graph
import os.path
import numpy as np
from mnist import MNIST

PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
RHO = 0.99
EPSILON = 1.0e-8

class RMSProp(le.Learning):
    def __init__(self):
        super(RMSProp, self).__init__()
        self.h1 = 0
        self.h2 = 0

    def fullyConnecterLayer(self, x, w, b):
        return np.dot(w, x) #+ b

    def renewParam(self, deltaW1, deltaB1, deltaW2, deltaB2):
        dW1, self.h1 = self.renewRMSP(self.h1, deltaW1)
        dW2, self.h2 = self.renewRMSP(self.h2, deltaW2)
        self.w1 -= dW1
        self.w2 -= dW2

    def renewRMSP(self, h, deltaW):
        h = RHO * h + (1.0 - RHO) * deltaW * deltaW
        dW = ETA / (np.sqrt(h) + EPSILON) * deltaW
        return (dW, h)



if __name__ == '__main__':
    l = RMSProp()
    graph = Graph()
    count = 0
    precision = 0
    inputX1 = np.empty((l.X_SIZE, B_SIZE))
    inputX2 = np.empty((M_SIZE, B_SIZE))
    deltaA = np.empty((CLASS_SIZE, B_SIZE))

    for count in xrange(l.N / B_SIZE * l.Epoch_Size):
        minibatch = np.random.choice(l.N, B_SIZE)
        averageOfEntropy = 0
        correct = 0
        j = 0
        for i in minibatch:
            inputX = l.X[i] / 256.0
            x, y1, y2,= l.forward(inputX)
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
            testres = l.test()
            print count / (l.N / B_SIZE)
            print averageOfEntropy
            print precision
            print testres
            graph.graphAppend(count / (l.N / B_SIZE), np.sum(averageOfEntropy), precision, testres)
            precision = 0
        count += 1
    graph.plot()
