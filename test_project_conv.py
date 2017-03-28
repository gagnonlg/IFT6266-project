import logging
import os
import subprocess

import h5py as h5
import PIL.Image
import numpy as np
import theano.tensor as T

import dataset
import network

fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
#logging.basicConfig(level='DEBUG', format=fmt)
logging.basicConfig(level='INFO', format=fmt)
log = logging.getLogger('test_project')

conv = network.Network()
# convolve into 20 maps of size
# input shape - filter shape + 1 = 64 - 5 + 1 = 60
# subsample by 2 => 20 30x30 maps
conv.add(network.Convolution(20, 3, 5, 5, border_mode='valid')) 
conv.add(network.MaxPool((2,2), ignore_border=True)) # output (batch, 20, 16, 16)
conv.add(network.Tanh())
# convolve into 50 maps of size
# input shape - filter shape + 1 = 30 - 5 + 1 = 26
# subsample by 2 => 50 13x13 maps
conv.add(network.Convolution(50, 20, 5, 5, border_mode='valid'))
conv.add(network.MaxPool((2,2), ignore_border=True))
conv.add(network.Tanh())
# flatten into (batch_size, 50 * 13 * 13)
conv.add(network.Flatten())
# Fully connected layers
conv.add(network.LinearTransformation((50*13*13, 1000)))
conv.add(network.Tanh())
conv.add(network.LinearTransformation((1000, 32*32*3)))

conv.compile(
    lr=0.0000001,
    momentum=0.5,
    batch_size=256,
    cache_size=(2560, (3, 64, 64), (32*32*3,)),
    vartype=(T.tensor4, T.matrix),
)

datapath = os.getenv('DATAPATH')
if datapath is None:
    datapath = os.getenv('PWD') + '/conv_dataset.h5'
log.info('datapath: %s', datapath)
h5dataset = h5.File(datapath, 'r')

# test figure
xt = h5dataset['val/input'][0]
yt = h5dataset['val/target'][0]

imgdir = '{}/test_images/{}'.format(os.getenv('HOME'), os.path.basename(os.getenv('LSCRATCH')))
subprocess.call(['mkdir', '-p', imgdir])

for i in range(1000):
    log.debug('xt.shape: %s', str(xt.shape))
    log.debug('xt[np.newaxis,:].shape: %s', str(xt[np.newaxis,:].shape))
    b = conv(xt[np.newaxis, :])
    img = dataset.reconstruct_from_unflat(xt, b[0])
    PIL.Image.fromarray(img.astype(np.uint8)).save('test_image_{}.jpg'.format(i))
    subprocess.call(['cp', 'test_image_{}.jpg'.format(i), imgdir])
    log.info('epoch %d', i)
    conv.train(
        X=h5dataset['train/input'],
        Y=h5dataset['train/target'],
        val_data=(h5dataset['val/input'], h5dataset['val/target']),
        n_epochs=1,
        start_epoch=i
    )

conv.save('test_network_conv.h5')
