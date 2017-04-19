import logging
logging.basicConfig(level='INFO')

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

trainX = gen_data(1280 * 100)

print '-> Defining discriminator'
discriminator = network.Network(is_GAN_discriminator=True)
discriminator.add(network.LinearTransformation((2, 100)))
discriminator.add(network.ReLU())
discriminator.add(network.LinearTransformation((100, 1)))
discriminator.add(network.Sigmoid())

print '-> Defining generator'
generator = network.Network()
generator.add(network.LinearTransformation((100, 10)))
generator.add(network.ReLU())
generator.add(network.LinearTransformation((10, 2)))

print '-> Compiling discriminator'

def discriminator_GAN_loss(x, y):
    return T.mean(T.log(x) + T.log(1 - y))

discriminator.compile(
    batch_size=128,
    cache_size=(1280, 2, 2),
    loss=discriminator_GAN_loss,
    use_ADAM=False
)

print '-> Compiling the generator'

def generator_GAN_loss(x, y):
    return T.mean(T.log(discriminator.expression(x)))

generator.compile(
    batch_size=128,
    cache_size=(128, 100, 2),
    loss=generator_GAN_loss,
    use_ADAM=False,
)

for epoch in range(10):
    for i in range(int(trainX.shape[0] / 1280.0)):
        X = trainX[i*1280:(i+1)*1280].astype('float32')
        Y = generator(np.random.uniform(size=(1280, 100)))

        discriminator.train(
            X=X,
            Y=Y,
            val_data=(X,Y),
            n_epochs=1
        )

        X = np.random.uniform(size=(128, 100)).astype('float32')

        generator.train(
            X=X,
            Y=generator(X),
            val_data=(X, generator(X)),
            n_epochs=1,
        )

    print 'epoch %d' % epoch
    print generator(X[:10])

    


