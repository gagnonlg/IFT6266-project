""" Simple regression test for the Densely connected MLP """

import logging
logging.basicConfig(level='INFO')

import matplotlib.pyplot as plt
import numpy as np

import mlp

## Generate the test data
print('-> Generating the data')

# random generator seed
np.random.seed(750)

# number of training points
N_TRAIN = 100000

# number of test points
N_TEST = 1000

# variance of additive noise
NOISE_STDDEV = 0.1

# path of training sample
PATH_TRAIN = 'sine.training.txt'

# path of test sample
PATH_TEST = 'sine.test.txt'

def __gen_data(n, path):
    data = np.zeros((n,2), dtype='float32')
    data[:,0] = np.random.uniform(0, 2*np.pi, n)
    data[:,1] = np.sin(data[:,0]) + np.random.normal(scale=NOISE_STDDEV, size=n)
    return data[:,0], data[:,1]

datax, datay = __gen_data(N_TRAIN, PATH_TRAIN)
testx, testy  = __gen_data(N_TEST, PATH_TEST)

mean = np.mean(datax)
std = np.std(datax)
datax -= mean
datax /= std
testx -= mean
testx /= std

datax = np.atleast_2d(datax).T
datay = np.atleast_2d(datay).T
testx = np.atleast_2d(testx).T

#create and train the network
print('-> Building the model')

network = mlp.MLP([1, 300, 1])
#training = network.training_function()

print('-> Training the model')

network.train(datax, datay, 50)

# batch_size=128
# n_train_batches=datax.size/batch_size
 
# nepochs = 50
# epoch = 0
# while (epoch < nepochs):
    
#     losses = np.zeros(n_train_batches)

#     for minibatch_index in range(n_train_batches):
#         i0 = minibatch_index * batch_size
#         i1 = (minibatch_index + 1) * batch_size
#         losses[minibatch_index] = training(datax[i0:i1], datay[i0:i1])

#     print 'epoch {}: avg. loss = {}'.format(epoch, np.mean(losses))
#     epoch += 1

print('-> Testing the model')
    
#test = network.test_function()
#outputs = test(testx)

# graph of result

isort = np.argsort(testx[:,0])
x = testx[isort] * std + mean
y = testy[isort]

# print np.min(x), np.max(x)

# print test(testx[isort])

plt.plot(x, network(testx[isort]), 'k')
plt.plot(x, np.sin(x), 'k--')
plt.scatter(x, y, facecolors='none', edgecolor='r')
plt.savefig('sine_test.png')
