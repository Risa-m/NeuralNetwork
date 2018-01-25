

import os.path
import numpy as np
from mnist import MNIST
import cPickle
with open("le4nn/le4MNIST_X.dump","rb") as f:
    X = cPickle.load(f)
    X = X.reshape((X.shape[0], 784))

PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01


class Recognition(Learning):
    def __init__(self):
        super(Recognition, self).__init__()
        self.data = X

    def recognize(self):
        correct = 0
        for i in xrange(Learning.p.shape[0]):
            inputX = Learning.p[i] / 256.0
            x, y1, y2 = self.forward(inputX)
            correct = correct + (1.0 / Learning.p.shape[0]) if(self.recogRes(y2)==Learning.q[i]) else correct
        return correct
