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
RHO = 5 # M_SIZE * 0.1

class DropOut(le.Learning):
    def __init__(self):
        super(DropOut, self).__init__()

    def dropout(self, x):
        a = np.array([0] * RHO + [1] * (x.shape[0] - RHO))
        np.random.shuffle(a)
        a = a.reshape(-1, 1)
        t = x * a
        return (t, a)

    def forward(self, x):
        layer1Out = self.inputLayer(x)
        layer2In = self.fullyConnecterLayer(layer1Out, self.w1, self.b1)
        layer2Out, a = self.dropout(layer2In)
        layer3In = self.fullyConnecterLayer(layer2Out, self.w2, self.b2)
        layer3Out = self.softmax(layer3In)
        return (layer1Out, layer2Out, layer3Out, a)

    def forwardtest(self, x):
        layer1Out = self.inputLayer(x)
        layer2In = self.fullyConnecterLayer(layer1Out, self.w1, self.b1)
        layer2Out = layer2In * (1 - RHO / M_SIZE)
        layer3In = self.fullyConnecterLayer(layer2Out, self.w2, self.b2)
        layer3Out = self.softmax(layer3In)
        return (layer1Out, layer2Out, layer3Out)


    def backPropagate(self, inputX1, inputX2, deltaA, deltaDrop):
        deltaX2, deltaW2, deltaB2 = self.backOfConnect(inputX2, self.w2, deltaA)
        _, deltaW1, deltaB1 = self.backOfConnect(inputX1, self.w1, deltaDrop * deltaX2)
        return (deltaW1, deltaB1, deltaW2, deltaB2)

    def test(self):
        correct = 0
        for i in range(DropOut.p.shape[0]):
            inputX = DropOut.p[i] / 256.0
            x, y1, y2 = self.forwardtest(inputX)
            correct = correct + (1.0 / DropOut.p.shape[0]) if(self.recogRes(y2)==DropOut.q[i]) else correct
        return correct


if __name__ == '__main__':
    l = DropOut()
    count = 0
    precision = 0
    while count <= (l.N / B_SIZE * 10) :
        minibatch = np.random.choice(l.N, B_SIZE)
        averageOfEntropy = 0
        inputX1 = np.zeros(l.X_SIZE).reshape(-1, 1)
        inputX2 = np.zeros(M_SIZE).reshape(-1, 1)
        deltaA = np.zeros(CLASS_SIZE).reshape(-1, 1)
        deltaDrop = np.zeros(M_SIZE).reshape(-1, 1)
        correct = 0
        for i in minibatch:
            inputX = l.X[i] / 256.0
            x, y1, y2, a = l.forward(inputX)
            ansY = [0] * l.Y[i] + [1] + [0] * (10 - l.Y[i] - 1)
            averageOfEntropy += l.crossEntropy(ansY, y2) / B_SIZE
            deltaDrop = np.hstack((deltaDrop, a))
            deltaA = np.hstack((deltaA, l.backOfSoftAndCross(ansY, y2)))
            inputX1 = np.hstack((inputX1, x))
            inputX2 = np.hstack((inputX2, y1))
            correct = correct + (1.0 / B_SIZE) if(l.recogRes(y2) == l.Y[i]) else correct
        deltaA = np.delete(deltaA, 0, axis=1)
        deltaDrop = np.delete(deltaDrop, 0, axis=1)
        inputX2 = np.delete(inputX2, 0, axis=1)
        inputX1 = np.delete(inputX1, 0, axis=1)

        deltaW1, deltaB1, deltaW2, deltaB2 = l.backPropagate(inputX1, inputX2, deltaA, deltaDrop)
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
