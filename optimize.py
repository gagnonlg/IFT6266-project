import logging
import os
import subprocess
import sys

sys.setrecursionlimit(50000)

import h5py as h5
import PIL.Image
import numpy as np

import dataset
import network

log = logging.getLogger('randomopt')
fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
logging.basicConfig(level='INFO', format=fmt)

#### some fixed parameters
early_stopping_patience = 10
improvement_threshold = 0.999
lr_reduce_cooldown = 2
batch_size = 256

#### pick hyperparameters
lr = np.random.uniform(1e-7, 1e-3)
l2 = np.random.uniform(0, 1e-3)
n_hidden_units = np.random.randint(200, 2000)
n_hidden_layer = np.random.randint(2, 7)
momentum = np.random.uniform(0, 1)
batch_norm = np.random.choice([True, False])
lr_reduce_patience = np.random.randint(3)
lr_reduce_factor = np.random.choice([1, np.random.uniform(0.1, 0.5)])
dropout_visible = np.random.choice([True, False])
dropout_hidden = np.random.choice([True, False])

### print hyperparameters to log
log.info('lr:%f', lr)
log.info('l2:%f', l2)
log.info('n_hidden_units:%d', n_hidden_units)
log.info('n_hidden_layer:%d', n_hidden_layer)
log.info('momentum:%f', momentum)
log.info('batch_norm:%d', batch_norm)
log.info('batch_size:%d', batch_size)
log.info('lr_reduce_patience:%d', lr_reduce_patience)
log.info('lr_reduce_factor:%f', lr_reduce_factor)
log.info('dropout_visible:%d', dropout_visible)
log.info('dropout_hidden:%d', dropout_hidden)

### build model
n_in = (64*64 - 32*32) * 3
n_out = 32*32*3
netw = network.Network()

structure = [n_in] + ([n_hidden_units] * n_hidden_layer) + [n_out]
n_in_out = zip(structure[:-1], structure[1:])
relu = ([True] * (len(n_in_out) - 1)) + [False]
batchnorm = [batch_norm] * len(n_in_out)

dropout = [0.5 if dropout_hidden else False] * len(n_in_out)
dropout[0] = 0.2 if dropout_visible else False

for ((nin,nout), relu, batchn, drp) in zip(n_in_out, relu, batchnorm, dropout):
    if batchn:
        netw.add(network.BatchNorm(nin))
    if drp:
        netw.add(network.Dropout(drp))
    netw.add(network.LinearTransformation((nin, nout), l2=l2))
    if relu:
        netw.add(network.ReLU())

netw.add(network.Sigmoid())
netw.add(network.ScaleOffset(scale=255.0))

netw.compile(
    lr=lr,
    momentum=momentum,
    batch_size=batch_size,
    cache_size=(20000, n_in, n_out)
)

###

datapath = os.getenv('DATAPATH')
if datapath is None:
    datapath = os.getenv('PWD') + '/mlp_dataset.h5'
log.info('datapath: %s', datapath)
h5dataset = h5.File(datapath, 'r')

# test figure
xt = h5dataset['val/input'][0]
yt = h5dataset['val/target'][0]

imgdir = '{}/test_images/{}'.format(os.getenv('HOME'), os.path.basename(os.getenv('LSCRATCH')))
subprocess.call(['mkdir', '-p', imgdir])

netw.train(
    X=h5dataset['train/input'],
    Y=h5dataset['train/target'],
    val_data=(h5dataset['val/input'], h5dataset['val/target']),
    n_epochs=1000,
    early_stopping_patience=early_stopping_patience,
    improvement_threshold=improvement_threshold,
    lr_reduce_factor=lr_reduce_factor,
    lr_reduce_patience=lr_reduce_patience,
    lr_reduce_cooldown=lr_reduce_cooldown,
    save_path='model.gz'
)

subprocess.call(['cp', 'model.gz', imgdir])

# tests
for i in range(0, 100):
    xt = h5dataset['val/input'][i]
    yt = h5dataset['val/target'][i]
    b = netw(xt[np.newaxis, :])
    img = dataset.reconstruct_from_flat(xt, b[0])
    PIL.Image.fromarray(img.astype(np.uint8)).save('test_image_{}.jpg'.format(i))

