import logging
import os
import subprocess

import h5py as h5
import PIL.Image
import numpy as np
import theano
import theano.tensor as T

import dataset
import network

fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
logging.basicConfig(level='INFO', format=fmt)
log = logging.getLogger('test_project')

########################################################################
log.info('Loading dataset')

home = os.getenv('HOME')
D_dataset = h5.File(home + '/D_dataset.h5', 'r')
G_dataset = h5.File(home + '/G_dataset.h5', 'r')

def D_data_gen(dset):

    def __gen(size):

        xdset = D_dataset[dset]

        while True:
            for idx in network.grouper(range(xdset.shape[0]), size):
                idx_ = filter(lambda n: n is not None, idx)
                i0 = idx_[0]
                i1 = idx_[-1]
                xbatch = xdset[i0:i1]
                xbatch *= (2.0 / 255.0)
                xbatch -= 1.0
                ybatch = np.random.uniform(0.7, 1.2, size=(size, 1)).astype('float32')
                yield xbatch, ybatch

    return __gen

def G_data_gen(dset):

    def __gen(size):

        xdset = G_dataset[dset]

        while True:
            for idx in network.grouper(range(xdset.shape[0]), size):
                idx_ = filter(lambda n: n is not None, idx)
                i0 = idx_[0]
                i1 = idx_[-1]
                xbatch = xdset[i0:i1]
                xbatch *= (2.0 / 255.0)
                xbatch -= 1.0
                ybatch = np.random.uniform(0.0, 0.3, size=(size, 1)).astype('float32')
                yield xbatch, ybatch

    return __gen


########################################################################
log.info('Building model')

n_z = 2000

n_in = (64*64 - 32*32) * 3
n_out = 32*32*3

gnetw = network.Network(paint_center=True)
gnetw.add(network.Generator())
gnetw.add(network.Convolution(256, 4, 5, 5, border_mode='half', strides=(2,2), gaus_init=True))
gnetw.add(network.BatchNorm(256, conv=True))
gnetw.add(network.ReLU())
gnetw.add(network.Convolution(128, 256, 5, 5, border_mode='half', gaus_init=True))
gnetw.add(network.BatchNorm(128, conv=True))
gnetw.add(network.ReLU())
gnetw.add(network.Convolution(32, 128, 5, 5, border_mode='half', gaus_init=True))
gnetw.add(network.BatchNorm(32, conv=True))
gnetw.add(network.ReLU())
gnetw.add(network.Convolution(3, 32, 5, 5, border_mode='half', gaus_init=True))
gnetw.add(network.Tanh())

dnetw = network.Network()
dnetw.add(network.Convolution(128, 3, 5, 5, border_mode='half', strides=(2,2), gaus_init=True))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.Convolution(64, 128, 5, 5, border_mode='half', strides=(2,2), gaus_init=True))
dnetw.add(network.BatchNorm(64,conv=True))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.Convolution(32, 64, 5, 5, border_mode='half', strides=(2,2), gaus_init=True))
dnetw.add(network.BatchNorm(32,conv=True))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.Flatten())
dnetw.add(network.LinearTransformation((32*8*8,1)))
dnetw.add(network.BatchNorm(32*8*8))
dnetw.add(network.Sigmoid())

dnetw.compile(
    lr=0.0002,
    ADAM_velocity=(0.5, 0.999),
    batch_size=256,
    cache_size=(256, (3, 64, 64), 1),
    use_ADAM=True,
    loss=network.binary_cross_entropy_loss,
    vartype=(T.tensor4, T.matrix)
)

def generator_GAN_loss(x, y):
    # y == ones, fed by the train_GAN function
    return network.binary_cross_entropy_loss(dnetw.expression(x), y)

gnetw.compile(
    lr=0.0002,
    ADAM_velocity=(0.5, 0.999),
    batch_size=256,
    cache_size=(256, (3, 64, 64), 1),
    use_ADAM=True,
    loss=generator_GAN_loss,
    vartype=(T.tensor4, T.matrix)
)

########################################################################
log.info('Training model')

network.train_GAN(
    G=gnetw,
    D=dnetw,
    batch_size=256,
    k_steps=1,
    n_epochs=50,
    steps_per_epoch=(D_dataset['train'].shape[0] / 256),
    data_gen=(D_data_gen('train'), D_data_gen('val')),
    z_prior_gen=(G_data_gen('train'), G_data_gen('val')),
    G_savepath='model_14.generator.best.h5',
    D_savepath='model_14.discriminator.best.h5'
)

gnetw.save('model_14.generator.final.h5')
dnetw.save('model_14.discriminator.final.h5')
