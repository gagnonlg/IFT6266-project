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
discriminator.add(network.LinearTransformation((100, 1)))
discriminator.add(network.Sigmoid())

print '-> Defining generator'
generator = network.Network()
generator.add(network.LinearTransformation((10, 100)))
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
    lr=0.01,
    batch_size=32,
    cache_size=(32, 10, 2),
    loss=generator_GAN_loss,
    use_ADAM=False,
)

for epoch in range(100):
    for i in range(int(trainX.shape[0] / 320.0)):
        X = trainX[i*320:(i+1)*320].astype('float32')
        Y = generator(np.random.uniform(size=(320, 10)))

        discriminator.train(
            X=X,
            Y=Y,
            val_data=(X,Y),
            n_epochs=1
        )

        X = np.random.uniform(size=(32, 10)).astype('float32')

        generator.train(
            X=X,
            Y=generator(X),
            val_data=(X, generator(X)),
            n_epochs=1,
        )

    print 'epoch %d' % epoch
    print generator(X[:10])

    
plt.scatter(trainX[:100,0], trainX[:100,1], color='b')
gx = generator(np.random.uniform(size=(100,10)))
plt.scatter(gx[:,0], gx[:,1], color='r')
plt.savefig('gan_test.png')
