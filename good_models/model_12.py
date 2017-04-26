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

datapath = os.getenv('DATAPATH')
if datapath is None:
    datapath = os.getenv('PWD') + '/mlp_dataset.h5'
log.info('datapath: %s', datapath)
h5dataset = h5.File(datapath, 'r')

vdataset = (
    h5dataset['val/input'][:1024],
    h5dataset['val/target'][:1024]
)

# test figure
xt = h5dataset['val/input'][0]
yt = h5dataset['val/target'][0]

imgdir = '{}/test_images/{}'.format(
    os.getenv('HOME'),
    "",#os.path.basename(os.getenv('LSCRATCH'))
)
subprocess.call(['mkdir', '-p', imgdir])


########################################################################
log.info('Building model')

n_z = 2000

n_in = (64*64 - 32*32) * 3
n_out = 32*32*3

gnetw = network.Network(copy_input=(0,n_in))
gnetw.add(network.BatchNorm(n_in + n_z))
gnetw.add(network.LinearTransformation((n_in + n_z, 2000)))
gnetw.add(network.ReLU(alpha=0.2))
gnetw.add(network.BatchNorm(2000))
gnetw.add(network.Dropout(0.5, at_test_time=True))
gnetw.add(network.LinearTransformation((2000, 2000)))
gnetw.add(network.ReLU(alpha=0.2))
gnetw.add(network.BatchNorm(2000))
gnetw.add(network.Dropout(0.5, at_test_time=True))
gnetw.add(network.LinearTransformation((2000, 2000)))
gnetw.add(network.ReLU(alpha=0.2))
gnetw.add(network.BatchNorm(2000))
gnetw.add(network.Dropout(0.5, at_test_time=True))
gnetw.add(network.LinearTransformation((2000, 2000)))
gnetw.add(network.ReLU(alpha=0.2))
gnetw.add(network.BatchNorm(2000))
gnetw.add(network.Dropout(0.5, at_test_time=True))
gnetw.add(network.LinearTransformation((2000, 2000)))
gnetw.add(network.ReLU(alpha=0.2))
gnetw.add(network.BatchNorm(2000))
gnetw.add(network.Dropout(0.5, at_test_time=True))
gnetw.add(network.LinearTransformation((2000, 2000)))
gnetw.add(network.ReLU(alpha=0.2))
gnetw.add(network.BatchNorm(2000))
gnetw.add(network.LinearTransformation((2000, n_out)))
gnetw.add(network.Tanh())

dnetw = network.Network()
dnetw.add(network.BatchNorm(n_in + n_out))
dnetw.add(network.LinearTransformation((n_in + n_out, 1000)))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000)))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000)))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000)))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000)))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000)))
dnetw.add(network.ReLU(alpha=0.2))
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1)))
dnetw.add(network.Sigmoid())

dnetw.compile(
    batch_size=1024,
    cache_size=(1024, n_in + n_out, 1),
    use_ADAM=False,
    loss=network.binary_cross_entropy_loss
)

def generator_GAN_loss(x, y):
    # y == ones, fed by the train_GAN function
    return network.binary_cross_entropy_loss(dnetw.expression(x), y)

gnetw.compile(
    batch_size=1024,
    cache_size=(1024, n_in + n_z, n_in + n_out),
    use_ADAM=True,
    loss=generator_GAN_loss
)

########################################################################
log.info('Training model')

def data_gen(dset):

    def __gen(size):

        xdset = h5dataset[dset+'/input']
        ydset = h5dataset[dset+'/target']

        while True:
            for idx in network.grouper(range(xdset.shape[0]), size):
                idx_ = filter(lambda n: n is not None, idx)
                i0 = idx_[0]
                i1 = idx_[-1]
                xbatch = np.concatenate([xdset[i0:i1], ydset[i0:i1]], axis=1)
                xbatch *= (2.0 / 255.0)
                xbatch -= 1.0
                ybatch = np.random.uniform(0.7, 1.2, size=(size, 1)).astype('float32')
                yield xbatch, ybatch

    return __gen

def z_prior_gen(dset):

    def __gen(size):

        xdset = h5dataset[dset+'/input']

        while True:
            for idx in network.grouper(range(xdset.shape[0]), size):
                idx_ = filter(lambda n: n is not None, idx)
                i0 = idx_[0]
                i1 = idx_[-1]
                subt = xdset[i0:i1]
                subt *= (2.0 / 255.0)
                subt -= 1.0
                zs = np.random.normal(size=(subt.shape[0], n_z)).astype(
                    'float32'
                )
                xbatch = np.concatenate([subt, zs], axis=1)
                ybatch = np.random.uniform(0.0, 0.3, size=(size, 1)).astype('float32')
                yield xbatch, ybatch

    return __gen


network.train_GAN(
    G=gnetw,
    D=dnetw,
    batch_size=1024,
    k_steps=1,
    n_epochs=50,
    steps_per_epoch=(h5dataset['train/input'].shape[0] / 1024),
    data_gen=(data_gen('train'), data_gen('val')),
    z_prior_gen=(z_prior_gen('train'), z_prior_gen('val')),
    G_savepath='model_12.generator.best.h5',
    D_savepath='model_12.discriminator.best.h5'
)

gnetw.save('model_12.generator.final.h5')
dnetw.save('model_12.discriminator.final.h5')
