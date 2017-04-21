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
gnetw.add(network.LinearTransformation((n_in + n_z, 1000), l2=0.001))
gnetw.add(network.ReLU())
gnetw.add(network.BatchNorm(1000))
gnetw.add(network.LinearTransformation((1000, 1000), l2=0.001))
gnetw.add(network.ReLU())
gnetw.add(network.BatchNorm(1000))
gnetw.add(network.LinearTransformation((1000, 1000), l2=0.001))
gnetw.add(network.ReLU())
gnetw.add(network.BatchNorm(1000))
gnetw.add(network.LinearTransformation((1000, n_out), l2=0.001))
gnetw.add(network.Sigmoid())
gnetw.add(network.ScaleOffset(scale=255.0))

def discriminator_GAN_loss(x, y):
    # x = D(data border+center)
    # y = D(generated border+center)
    return - T.mean(T.log(x) + T.log(1 - y))

dnetw = network.Network(is_GAN_discriminator=True)
dnetw.add(network.BatchNorm(n_in + n_out))
dnetw.add(network.LinearTransformation((n_in + n_out, 1000), l2=0.001))
dnetw.add(network.ReLU())
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000), l2=0.001))
dnetw.add(network.ReLU())
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1000), l2=0.001))
dnetw.add(network.ReLU())
dnetw.add(network.BatchNorm(1000))
dnetw.add(network.LinearTransformation((1000, 1), l2=0.001))
dnetw.add(network.Sigmoid())
dnetw.compile(
    batch_size=1024,
    cache_size=(1024, n_in + n_out, 1),
    use_ADAM=True,
    loss=discriminator_GAN_loss,
)

def generator_GAN_loss(x, y):
    return - T.mean(T.log(dnetw.expression(x)))

gnetw.compile(
    batch_size=1024,
    cache_size=(1024, n_in + n_z, n_in + n_out),
    use_ADAM=True
)

########################################################################
log.info('Training model')

def z_prior_gen(size):
    xdset = h5dataset['train/input']
    
    while True:
        for idx in network.grouper(range(xdset.shape[0]), size):
            idx_ = filter(lambda n: n is not None, idx)
            i0 = idx_[0]
            i1 = idx_[-1]
            zs = np.random.uniform(size=(size, n_z)).astype('float32')
            yield np.concatenate([xdset[i0:i1], zs], axis=1)

def v_z_prior_gen(size):
    xdset = h5dataset['val/input']
    
    while True:
        for idx in network.grouper(range(xdset.shape[0]), size):
            idx_ = filter(lambda n: n is not None, idx)
            i0 = idx_[0]
            i1 = idx_[-1]
            zs = np.random.uniform(size=(size, n_z)).astype('float32')
            yield np.concatenate([xdset[i0:i1], zs], axis=1)



def data_gen(dset):

    def __gen(size):
    
        xdset = h5dataset[dset+'/input']
        ydset = h5dataset[dset+'/target']
             
        while True:
            for idx in network.grouper(range(xdset.shape[0]), size):
                idx_ = filter(lambda n: n is not None, idx)
                i0 = idx_[0]
                i1 = idx_[-1]
                yield np.concatenate([xdset[i0:i1], ydset[i0:i1]], axis=1)
                
    return __gen

def z_prior_gen(dset):

    def __gen(size):
    
        xdset = h5dataset[dset+'/input']
             
        while True:
            for idx in network.grouper(range(xdset.shape[0]), size):
                idx_ = filter(lambda n: n is not None, idx)
                i0 = idx_[0]
                i1 = idx_[-1]
                zs = np.random.uniform(size=(i1-i0, n_z)).astype('float32')
                yield np.concatenate([xdset[i0:i1], zs], axis=1)
                
    return __gen

            
network.train_GAN(
    G=gnetw,
    D=dnetw,
    batch_size=1024,
    k_steps=1,
    n_epochs=100,
    steps_per_epoch=(h5dataset['train/input'].shape[0] / 1024),
    data_gen=(data_gen('train'), data_gen('val')),
    z_prior_gen=(z_prior_gen('train'), z_prior_gen('val')),
    G_savepath='model_07.generator.h5',
    D_savepath='model_07.discriminator.h5'
)
