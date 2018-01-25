import Learning as le
import ReLU
import Pooling as po
import Convolution as co
import numpy as np
from mnist import MNIST
import os.path



PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 10000
ETA = 0.01
np.random.seed(200)


def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def backOfConnect(x, w, deltaY):
    deltaX = np.dot(w.T, deltaY)
    deltaW = np.dot(deltaY, x.T)
    deltaB = (np.sum(deltaY, axis=1)).reshape(-1,1)
    return (deltaX, deltaW, deltaB)

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

p, q = mndata.load_testing()
p = np.array(p)
q = np.array(q)
p = p.reshape((p.shape[0], PICT_HEIGHT, PICT_WIDTH))

import cPickle
with open("le4nn/le4MNIST_X.dump","rb") as f:
    X = cPickle.load(f)
    X = X.reshape((X.shape[0], PICT_HEIGHT, PICT_WIDTH))

R = 3
k1 = 32
#k2 = 4
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


if(os.path.exists("convolution32-4.npz")):
    load_array = np.load("convolution32-4.npz")
    w1 = load_array["w1"]
    b1 = load_array["b1"]
    w2 = load_array["w2"]
    b2 = load_array["b2"]
    w3 = load_array["w3"]
    b3 = load_array["b3"]
    load_array.close()



precision = 0
batchset = np.empty((X.shape[1] , X.shape[2], B_SIZE))
"""
ansY = np.empty((CLASS_SIZE, B_SIZE))
answer = np.array([Y[i] for i in xrange(B_SIZE)])
for i in xrange(B_SIZE):
    ansY[:,i] = np.array([0] * answer[i] + [1] + [0] * (10 - answer[i] - 1))
"""
answer = [9,1,6,5,1,5,2,4,7,7,7,4,7,6,5,0,6,1,6,4,2,9,4,0,4,0,8,5,4,7,9,9,4,1,4,3,7,8,5,5,0,5,0,8,9,3,8,8,0,1,8,6,5,0,4,2,1,7,3,7,6,2,5,9,5,0,7,4,7,2,9,7,1,1,2,1,4,8,4,2,1,7,4,0,2,9,4,5,1,0,6,7,4,3,7,0,1,3,2,1,1,1,7,0,7,7,8,3,3,7,6,5,1,4,7,7,7,9,2,1,1,2,8,0,8,2,4,8,4,7,5,4,3,6,5,3,6,3,6,7,9,8,0,1,5,2,6,6,1,6,3,1,6,4,5,5,5,4,9,0,7,3,6,4,5,4,8,5,8,6,8,3,7,9,7,6,0,0,2,0,0,4,2,9,0,8,0,6,1,2,4,3,1,0,4,8,5,7,8,0,3,6,5,5,4,8,6,5,5,8,9,4,6,3,7,2,3,1,3,3,7,7,4,0,1,2,5,3,8,8,5,7,4,8,8,8,8,7,4,6,8,7,6,8,0,2,2,4,4,7,3,7,9,2,7,9,5,0,6,8,9,0,1,5,2,3,6,1,1,2,9,9,3,4,9,9,6,7,7,8,2,4,5,9,9,4,5,7,7,3,0,5,5,0,8,5,2,2,3,3]

j = 0
for i in xrange(B_SIZE):
    batchset[:,:,j] = X[i]
    j += 1

x1, y1 = c1.convolution(batchset, w1, b1)
z1 = relu.relu(y1)


x2 = np.empty((k1 * PICT_HEIGHT * PICT_WIDTH, B_SIZE))
t = np.split(z1, B_SIZE, axis=1)
for i in xrange(B_SIZE):
    x2[:,i] = t[i].ravel()


y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * B_SIZE)))
x3 = relu.relu(y2)
y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * B_SIZE)))
res = softmax(y4)

#averageOfEntropy = np.sum(crossEntropy(ansY, res)) / B_SIZE
recog = np.argmax(res, axis=0)
precision = len(np.where(answer - recog[0:300] == 0)[0]) * 1.0 / 300
#print averageOfEntropy
#print recog[50:61]
#print answer
print precision
"""
result = recog.tolist()
f = open('result.txt', 'w')
for x in result:
    f.write(str(x) + "\n")
f.close()
"""
"""
import matplotlib.pyplot as plt
from pylab import cm
i = 50
ppp = np.hstack(((X[i+0], X[i+1], X[i+2], X[i+3], X[i+4], X[i+5], X[i+6], X[i+7],X[i+8], X[i+9], X[i+10])))
plt.imshow(ppp, cmap=cm.gray)
plt.show()
"""
