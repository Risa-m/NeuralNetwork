import Learning as le
import os.path
import numpy as np
from mnist import MNIST

PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
RHO = 0.95
EPSILON = 1.0e-6

class AdaDelta(le.Learning):
    def __init__(self):
        super(AdaDelta, self).__init__()
        self.h1 = 0
        self.h2 = 0
        self.s1 = 0
        self.s2 = 0

    def fullyConnecterLayer(self, x, w, b):
        return np.dot(w, x) #+ b

    def renewParam(self, deltaW1, deltaB1, deltaW2, deltaB2):
        self.w1 += self.renewAdaDelta(self.h1, self.s1, deltaW1)
        self.w2 += self.renewAdaDelta(self.h2, self.s2, deltaW2)
        """
        self.h1 = RHO * self.h1 + (1 - RHO) * deltaW1 * deltaW1
        dW1 = - np.sqrt(self.s1 + EPSILON) / np.sqrt(self.h1 + EPSILON) * deltaW1
        self.s1 = RHO * self.s1 + (1 - RHO) * dW1 * dW1
        self.w1 += dW1
#        self.b1 -= (deltaB1 * ETA)
        self.h2 += deltaW2 * deltaW2
        self.h2 = RHO * self.h2 + (1 - RHO) * deltaW2 * deltaW2
        dW2 = - np.sqrt(self.s2 + EPSILON) / np.sqrt(self.h2 + EPSILON) * deltaW2
        self.s2 = RHO * self.s2 + (1 - RHO) * dW2 * dW2
        self.w2 += dW2
#        self.b2 -= (deltaB2 * ETA)
        """
    def renewAdaDelta(self, h, s, deltaW):
        h = RHO * h + (1 - RHO) * deltaW * deltaW
        dW = - np.sqrt(s + EPSILON) / np.sqrt(h + EPSILON) * deltaW
        s = RHO * s + (1 - RHO) * dW * dW
        return dW



if __name__ == '__main__':
    l = AdaDelta()
    count = 0
    precision = 0
    while count <= (l.N / B_SIZE * 10) :
        minibatch = np.random.choice(l.N, B_SIZE)
        averageOfEntropy = 0
        inputX1 = np.zeros(l.X_SIZE).reshape(-1, 1)
        inputX2 = np.zeros(M_SIZE).reshape(-1, 1)
        deltaA = np.zeros(CLASS_SIZE).reshape(-1, 1)
        correct = 0
        for i in minibatch:
            inputX = l.X[i] / 256.0
            x, y1, y2,= l.forward(inputX)
            ansY = [0] * l.Y[i] + [1] + [0] * (10 - l.Y[i] - 1)
            averageOfEntropy += l.crossEntropy(ansY, y2) / B_SIZE
            deltaA = np.hstack((deltaA, l.backOfSoftAndCross(ansY, y2)))
            inputX1 = np.hstack((inputX1, x))
            inputX2 = np.hstack((inputX2, y1))
            correct = correct + (1.0 / B_SIZE) if(l.recogRes(y2) == l.Y[i]) else correct
        deltaA = np.delete(deltaA, 0, axis=1)
        inputX2 = np.delete(inputX2, 0, axis=1)
        inputX1 = np.delete(inputX1, 0, axis=1)

        deltaW1, deltaB1, deltaW2, deltaB2 = l.backPropagate(inputX1, inputX2, deltaA)
        l.renewParam(deltaW1, deltaB1, deltaW2, deltaB2)

        precision += correct / (l.N / B_SIZE)
        if (count % (l.N / B_SIZE)) == 0:
            testres = l.test()
            print count / (l.N / B_SIZE)
            print averageOfEntropy
            print precision
            print testres
            precision = 0
        count += 1
