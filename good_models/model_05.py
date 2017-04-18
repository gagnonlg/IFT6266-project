import logging
import os
import subprocess

import h5py as h5
import PIL.Image
import numpy as np

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
    os.path.basename(os.getenv('LSCRATCH'))
)
subprocess.call(['mkdir', '-p', imgdir])


########################################################################
log.info('Building model')

n_in = (64*64 - 32*32) * 3
n_out = 32*32*3
netw = network.Network()
netw.add(network.BatchNorm(n_in))
netw.add(network.LinearTransformation((n_in, 1000), l2=0.001))
netw.add(network.ReLU())
netw.add(network.BatchNorm(1000))
netw.add(network.LinearTransformation((1000, 1000), l2=0.001))
netw.add(network.ReLU())
netw.add(network.BatchNorm(1000))
netw.add(network.LinearTransformation((1000, 1000), l2=0.001))
netw.add(network.ReLU())
netw.add(network.BatchNorm(1000))
netw.add(network.LinearTransformation((1000, n_out), l2=0.001))
netw.add(network.Sigmoid())
netw.add(network.ScaleOffset(scale=255.0))
netw.compile(
    lr=0.0000001,
    batch_size=1024,
    cache_size=(10240, n_in, n_out),
    use_ADAM=True,
    loss=network.cross_entropy_vector_loss
)

########################################################################
log.info('Training model')

for i in range(1000):

    log.info('epoch %d: training pass', i)
    
    netw.train(
        X=h5dataset['train/input'],
        Y=h5dataset['train/target'],
        val_data=vdataset,
        n_epochs=1,
        start_epoch=i
    )

    log.info('epoch %d: testing pass', i)

    b = netw(xt[np.newaxis, :])
    img = dataset.reconstruct_from_flat(xt, b[0]).astype(np.uint8)
    PIL.Image.fromarray(img).save('test_image_{}.jpg'.format(i))
    subprocess.call(['cp', 'test_image_{}.jpg'.format(i), imgdir])

    # save snapshot every 100 epochs
    if (i % 100) == 0:
        log.info('epoch %d: saving model snapshot', i)
        netw.save('model_05.{}.h5'.format(i))

# final snapshot
log.info('Saving final model snapshot')
netw.save('model_05.final.h5')

