import logging
import numpy as np
import network
import matplotlib.pyplot as plt
import cPickle

logging.basicConfig(level='DEBUG')

print '-> Generating the data'
# random generator seed
np.random.seed(750)
# number of training points
N_TRAIN = 100000
# number of test points
N_TEST = 1000
# variance of additive noise
NOISE_STDDEV = 0.1

def __gen_data(n):
    data = np.zeros((n,2), dtype='float32')
    data[:,0] = np.random.uniform(0, 2*np.pi, n)
    data[:,1] = np.sin(data[:,0]) + np.random.normal(scale=NOISE_STDDEV, size=n)
    return data[:,0], data[:,1]

datax, datay = __gen_data(N_TRAIN)
testx, testy  = __gen_data(N_TEST)

mean = np.mean(datax)
std = np.std(datax)
datax -= mean
datax /= std
testx -= mean
testx /= std

datax = np.atleast_2d(datax).T
datay = np.atleast_2d(datay).T
testx = np.atleast_2d(testx).T

print '-> Building the model'

mlp = network.Network()
mlp.add(network.LinearTransformation((1, 300), l2=0.001))
mlp.add(network.ReLU())
mlp.add(network.LinearTransformation((300, 1), l2=0.001))
mlp.compile(lr=0.02, momentum=0.5)

print '-> Training the model'

mlp.train(datax, datay, 50, 128)
mlp.save('test_model_1.gz')

print '-> Testing the model'

mlp = network.Network.load('test_model_1.gz')
    
isort = np.argsort(testx[:,0])
x = testx[isort] * std + mean
y = testy[isort]

plt.plot(x, mlp(testx[isort]), 'k')
plt.plot(x, np.sin(x), 'k--')
plt.scatter(x, y, facecolors='none', edgecolor='r')
plt.savefig('sine_test.png')
