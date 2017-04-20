import logging
logging.basicConfig(level='WARNING')

import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T

import network

print '-> Generating data'
def gen_data(N):
    mean = [100.0, -100.0]
    cov = [
        [1.00, 0.20],
        [0.20, 0.25]
    ]
    return np.random.multivariate_normal(mean, cov, N)

trainX = gen_data(32000)

print '-> Defining discriminator'
discriminator = network.Network(is_GAN_discriminator=True)
discriminator.add(network.LinearTransformation((2, 100)))
discriminator.add(network.ReLU())
discriminator.add(network.LinearTransformation((100, 100)))
discriminator.add(network.ReLU())
discriminator.add(network.LinearTransformation((100, 1)))
discriminator.add(network.Sigmoid())

print '-> Defining generator'
generator = network.Network()
generator.add(network.LinearTransformation((10, 100)))
generator.add(network.ReLU())
generator.add(network.LinearTransformation((100, 100)))
generator.add(network.ReLU())
generator.add(network.LinearTransformation((100, 2)))

print '-> Compiling discriminator'

def discriminator_GAN_loss(x, y):
    return - T.mean(T.log(x) + T.log(1 - y))

discriminator.compile(
    lr=0.001,
    batch_size=32,
    cache_size=(320, 2, 2),
    loss=discriminator_GAN_loss,
    use_ADAM=False
)

print '-> Compiling the generator'

def generator_GAN_loss(x, y):
    return - T.mean(T.log(discriminator.expression(x)))

generator.compile(
    #lr=0.01,
    batch_size=32,
    cache_size=(32, 10, 2),
    loss=generator_GAN_loss,
    use_ADAM=True,
)

def data_gen(size):
    while True:
        yield gen_data(size).astype('float32')

print '-> Training the GAN'
        
network.train_GAN(
    G=generator,
    D=discriminator,
    batch_size=32,
    k_steps=10,
    n_epochs=100,
    steps_per_epoch=100,
    data_gen=data_gen,
    v_data_gen=data_gen,
    z_prior=lambda s: np.random.uniform(size=(s,10)).astype('float32')
)
    
plt.scatter(trainX[:100,0], trainX[:100,1], color='b')
gx = generator(np.random.uniform(size=(100,10)))
plt.scatter(gx[:,0], gx[:,1], color='r')
plt.savefig('gan_test.png')
