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
ALPHA = 0.9

class MSDG(le.Learning):
    def __init__(self):
        super(MSDG, self).__init__()
        self.predeltaW1 = 0
        self.predeltaW2 = 0

    def fullyConnecterLayer(self, x, w, b):
        return np.dot(w, x) #+ b

    def renewParam(self, deltaW1, deltaB1, deltaW2, deltaB2):
        self.predeltaW1 = ALPHA * self.predeltaW1 - ETA * deltaW1
        self.w1 += self.predeltaW1
        self.b1 -= (deltaB1 * ETA)
        self.predeltaW2 = ALPHA * self.predeltaW2 - ETA * deltaW2
        self.w2 += self.predeltaW2
        self.b2 -= (deltaB2 * ETA)


if __name__ == '__main__':
    l = MSDG()
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
