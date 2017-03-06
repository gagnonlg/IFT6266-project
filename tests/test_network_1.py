import logging
import numpy as np
import network
import matplotlib.pyplot as plt
import cPickle

logging.basicConfig(level='INFO')

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

datax = np.atleast_2d(datax).T
datay = np.atleast_2d(datay).T
testx = np.atleast_2d(testx).T

print '-> Building the model'

mlp = network.Network()
mlp.add(network.BatchNorm(1))
mlp.add(network.LinearTransformation((1, 100), l2=0.00001))
mlp.add(network.ReLU())
mlp.add(network.BatchNorm(100))
mlp.add(network.LinearTransformation((100, 100), l2=0.00001))
mlp.add(network.ReLU())
mlp.add(network.LinearTransformation((100, 1), l2=0.0))
mlp.compile(lr=0.01, momentum=0.5, batch_size=256, cache_size=(1000,1,1))

print '-> Training the model'

mlp.train(datax, datay, 10)
mlp.save('test_model_1.gz')

print '-> Testing the model'

mlp = network.Network.load('test_model_1.gz')
    
isort = np.argsort(testx[:,0])
x = testx[isort]
y = testy[isort]

plt.plot(x, mlp(testx[isort]), 'k')
plt.plot(x, np.sin(x), 'k--')
plt.scatter(x, y, facecolors='none', edgecolor='r')
plt.savefig('sine_test.png')
