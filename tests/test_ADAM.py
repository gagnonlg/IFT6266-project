import logging
import numpy as np
import network
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import cPickle
import theano.tensor as T

logging.basicConfig(level='INFO')

print '-> Generating the data'
# random generator seed
seed=750
np.random.seed(seed)
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
    return np.atleast_2d(data[:,0]).T, np.atleast_2d(data[:,1]).T

datax, datay = __gen_data(N_TRAIN)
testx, testy  = __gen_data(N_TEST)
validx, validy  = __gen_data(N_TEST)

print '-> Building the model'
np.random.seed(35723)
#(x - mu) / std = (x + mmu) * stdi = x * stdi + mmu * stdi
mu = np.mean(datax)
std = np.std(datax)

#datax -= mu
#datax /= std
class Sigmoid(network.Layer):
    def expression(self, X):
        return T.nnet.sigmoid(X)

mlp = network.Network()
mlp.add(network.ScaleOffset(scale=1.0/std, offset=(- mu / std)))
mlp.add(network.LinearTransformation((1, 300)))
mlp.add(network.ReLU())
mlp.add(network.LinearTransformation((300, 1)))
mlp.compile(use_ADAM=True, batch_size=128, cache_size=(1280,1,1))

print '-> Training the model'

mlp.train(
    datax,
    datay,
    val_data=(validx, validy),
    n_epochs=50,
    savepath='test_model_3.h5'
)

print '-> Testing the model'

mlp.load('test_model_3.h5')

isort = np.argsort(testx[:,0])
x = testx[isort] 
y = testy[isort]

plt.plot(x, mlp(testx[isort]), 'k')
plt.plot(x, np.sin(x), 'k--')
plt.scatter(x, y, facecolors='none', edgecolor='r')
plt.savefig('sine_test.png')
