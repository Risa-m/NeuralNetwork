import Learning2 as le
from Graph import Graph
import os.path
import numpy as np
from mnist import MNIST

PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.001

class AdaGrad(le.Learning):
    def __init__(self):
        super(AdaGrad, self).__init__()
        self.h1 = 1.0e-8
        self.h2 = 1.0e-8


    def fullyConnecterLayer(self, x, w, b):
        return np.dot(w, x) #+ b

    def renewParam(self, deltaW1, deltaB1, deltaW2, deltaB2):
        self.h1 += deltaW1 * deltaW1
        self.w1 -= ETA / np.sqrt(self.h1) * deltaW1
        self.b1 -= (deltaB1 * ETA)
        self.h2 += deltaW2 * deltaW2
        self.w2 -= ETA / np.sqrt(self.h2) * deltaW2
        self.b2 -= (deltaB2 * ETA)


if __name__ == '__main__':
    l = AdaGrad()
    #Learning
    """
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
    """
    #Learning2
#    graph = Graph()
    count = 0
    precision = 0
    averageOfEntropy = 0
    inputX1 = np.empty((l.X_SIZE, B_SIZE))
    ansY = np.empty((CLASS_SIZE, B_SIZE))
    for count in xrange(l.N / B_SIZE * 50 + 1):
        choice = np.random.choice(l.N, B_SIZE)
        correct = 0
        j = 0
        answer = np.array([l.Y[i] for i in choice])
        for i in choice:
            inputX1[:,j] = l.X[i].ravel() / 256.0
            ansY[:,j] = np.array([0] * answer[j] + [1] + [0] * (10 - answer[j] - 1))
            j += 1
        x, y1, y2 = l.forward(inputX1)

        deltaW1, deltaB1, deltaW2, deltaB2 = l.backPropagate(x, y1, l.backOfSoftAndCross(ansY, y2))
        l.renewParam(deltaW1, deltaB1, deltaW2, deltaB2)
        recog = np.argmax(y2, axis=0)
        averageOfEntropy = np.sum(l.crossEntropy(ansY, y2)) / B_SIZE
        precision += len(np.where(answer - recog == 0)[0]) * 1.0 / l.N

        if (count % (l.N / B_SIZE)) == 0:
#            testres = l.test()
            print count / (l.N / B_SIZE)
            print averageOfEntropy
            print precision
#            print testres
#            graph.graphAppend(count / (l.N / B_SIZE), np.sum(averageOfEntropy), precision, testres)
            precision = 0
        count += 1
#    graph.plot()
