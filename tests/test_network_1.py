import logging
import numpy as np
import network
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

rng = T.shared_randomstreams.RandomStreams(1991)

mlp = network.Network()
mlp.add(network.BatchNorm(1))
mlp.add(network.LinearTransformation((1, 2000), l2=0.00000))
mlp.add(network.ReLU())
mlp.add(network.Dropout(0.5, rng=rng))
mlp.add(network.LinearTransformation((2000, 1), l2=0.00000))
mlp.compile(lr=0.01, momentum=0.0, batch_size=64, cache_size=(1000,1,1))

print '-> Training the model'

mlp.train(
    datax,
    datay,
    val_data=(validx, validy),
    n_epochs=100,
    early_stopping_patience=10,
    improvement_threshold=0.999,
    lr_reduce_factor=0.5,
    lr_reduce_patience=5,
    lr_reduce_cooldown=2,
    save_path='test_model_1.gz',
)

print '-> Testing the model'

mlp = network.Network.load('test_model_1.gz')
    
isort = np.argsort(testx[:,0])
x = testx[isort]
y = testy[isort]

plt.plot(x, mlp(testx[isort]), 'k')
plt.plot(x, np.sin(x), 'k--')
plt.scatter(x, y, facecolors='none', edgecolor='r')
plt.savefig('sine_test.png')
