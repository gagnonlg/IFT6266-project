import logging
import os
import subprocess

import h5py as h5
import PIL.Image
import numpy as np

import dataset
import network

fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
logging.basicConfig(level='DEBUG', format=fmt)
log = logging.getLogger('test_project')

n_in = (64*64 - 32*32) * 3
n_out = 32*32*3
netw = network.Network()
netw.add(network.ScaleOffset(scale=1.0/255))
netw.add(network.LinearTransformation((n_in, 1000)))
netw.add(network.ReLU())
netw.add(network.LinearTransformation((1000, 1000)))
netw.add(network.ReLU())
netw.add(network.LinearTransformation((1000, 1000)))
netw.add(network.ReLU())
netw.add(network.LinearTransformation((1000, n_out)))
netw.add(network.Sigmoid())
netw.add(network.ScaleOffset(scale=255.0))
netw.compile(
    lr=0.00001,
    momentum=0.5,
    batch_size=1000,
    cache_size=(20000, n_in, n_out)
)

datapath = os.getenv('DATAPATH')
if datapath is None:
    datapath = os.getenv('PWD') + '/mlp_dataset.h5'
log.info('datapath: %s', datapath)
h5dataset = h5.File(datapath, 'r')

# test figure
xt = h5dataset['val/input'][0]
yt = h5dataset['val/target'][0]

imgdir = '{}/test_images/{}'.format(os.getenv('HOME'), os.getenv('PBS_O_JOBID'))
subprocess.call(['mkdir', '-p', imgdir])

for i in range(1000):
    b = netw(xt[np.newaxis, :])
    img = dataset.reconstruct_from_flat(xt, b[0])
    PIL.Image.fromarray(img.astype(np.uint8)).save('test_image_{}.jpg'.format(i))
    subprocess.call(['cp', 'test_image_{}.jpg'.format(i), imgdir])
    log.info('epoch %d', i)
    netw.train(
        X=h5dataset['train/input'],
        Y=h5dataset['train/target'],
        n_epochs=1,
        start_epoch=i
    )

netw.save('test_network.h5')
