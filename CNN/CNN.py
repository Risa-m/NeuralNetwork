import Learning as le
import ReLU
import Pooling as po
import Convolution as co
import numpy as np
from mnist import MNIST
import time
import os.path

PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
ALPHA = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8

R = 3
k1 = 8
d = 2

np.random.seed(200)

mndata = MNIST("./le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
N = X.shape[0]
X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
Y = np.array(Y)

p, q = mndata.load_testing()
p = np.array(p)
q = np.array(q)
p = p.reshape((p.shape[0], PICT_HEIGHT, PICT_WIDTH))

w1 = np.random.normal(0.0, 1.0 / (R * R * k1 * PICT_WIDTH * PICT_HEIGHT), (k1, R * R))
b1 = np.random.normal(1.0, 1.0 / (R * R * k1), (k1, 1))

w2 = np.random.normal(0.0, 1.0 / (M_SIZE), (M_SIZE, PICT_HEIGHT*PICT_WIDTH*k1/(d*d)))#
b2 = np.random.normal(1.0, 1.0 / (M_SIZE), (M_SIZE, 1))

w3 = np.random.normal(0.0, 1.0 / (CLASS_SIZE), (CLASS_SIZE, M_SIZE))
b3 = np.random.normal(1.0, 1.0 / (CLASS_SIZE), (CLASS_SIZE, 1))

mw = [0,0,0]
vw = [0,0,0]
mb = [0,0,0]
vb = [0,0,0]
adamT = 0

def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def backOfConnect(x, w, deltaY):
    deltaX = np.dot(w.T, deltaY)
    deltaW = np.dot(deltaY, x.T)
    deltaB = (np.sum(deltaY, axis=1)/B_SIZE).reshape(-1,1)
    return (deltaX, deltaW, deltaB)

def softmax(a):
    alpha = np.max(a, axis=0)
    return np.exp(a - alpha) / np.sum(np.exp(a - alpha), axis=0)

def backOfSoftAndCross(ansY, y):
    return ((y - ansY) / B_SIZE)

def crossEntropy(AnsY, y):
    return np.sum(AnsY * np.log(y) * -1, axis=0)

def test():
    batchset = np.empty((p.shape[1] , p.shape[2], TEST_SIZE))
    answer = np.array([q[i] for i in xrange(TEST_SIZE)])
    for i in xrange(TEST_SIZE):
        batchset[:,:,i] = p[i]
    x1, y1 = c_test.convolution(batchset, w1, b1)
    z1 = relu.relu(y1)
    x2, choice1 = p_test.pooling(z1)
    y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * TEST_SIZE)))
    x3 = relu.relu(y2)
    y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * TEST_SIZE)))
    res = softmax(y4)
    recog = np.argmax(res, axis=0)
    precision = len(np.where(answer - recog == 0)[0]) * 1.0 / TEST_SIZE
    return precision

def renewAdam(m, v, deltaW, t):
    m = BETA1 * m + (1 - BETA1) * deltaW
    v = BETA2 * v + (1 - BETA2) * deltaW * deltaW
    bm = m / (1 - np.power(BETA1, t))
    bv = v / (1 - np.power(BETA2, t))
    dW = ALPHA * bm / (np.sqrt(bv) + EPSILON)
    return (dW, m, v)


relu = ReLU.ReLU()

c1 = co.Convolution(PICT_HEIGHT, PICT_WIDTH, B_SIZE, R, k1)
p1 = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, B_SIZE, k1)
TEST_SIZE = q.shape[0]

c_test = co.Convolution(PICT_HEIGHT, PICT_WIDTH, TEST_SIZE, R, k1)
p_test = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, TEST_SIZE, k1)

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
    x2, choice1 = p1.pooling(z1)
    y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * B_SIZE)))
    x3 = relu.sigmoid(y2)
    y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * B_SIZE)))
    res = softmax(y4)

    recog = np.argmax(res, axis=0)
    averageOfEntropy = np.sum(crossEntropy(ansY, res)) / B_SIZE
    precision += len(np.where(answer - recog == 0)[0]) * 1.0 / N
    if (count % (N / B_SIZE)) == 0:
        print count / (N / B_SIZE)
        print averageOfEntropy
        print precision
        precision = 0

    deltaSoft = backOfSoftAndCross(ansY, res)
    deltaX3, deltaW3, deltaB3 = backOfConnect(x3, w3, deltaSoft)
    deltaRelu2 = relu.backOfSig(x3, deltaX3)
    deltaX2, deltaW2, deltaB2 = backOfConnect(x2, w2, deltaRelu2)
    deltaPooling = p1.backOfPooling(deltaX2, choice1)
    deltaRelu1 = relu.backOfReLU(z1, deltaPooling)
    deltaX1, deltaW1, deltaB1 = backOfConnect(x1, w1, deltaRelu1)

    adamT += 1
    dW1, mw[0], vw[0] = renewAdam(mw[0], vw[0], deltaW1, adamT)
    dW2, mw[1], vw[1] = renewAdam(mw[1], vw[1], deltaW2, adamT)
    dW3, mw[2], vw[2] = renewAdam(mw[2], vw[2], deltaW3, adamT)
    dB1, mb[0], vb[0] = renewAdam(mb[0], vb[0], deltaB1, adamT)
    dB2, mb[1], vb[1] = renewAdam(mb[1], vb[1], deltaB2, adamT)
    dB3, mb[2], vb[2] = renewAdam(mb[2], vb[2], deltaB3, adamT)
    w1 -= dW1
    w2 -= dW2
    w3 -= dW3
    b1 -= dB1
    b2 -= dB2
    b3 -= dB3


#np.savez('CNNadam8-30s.npz', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
