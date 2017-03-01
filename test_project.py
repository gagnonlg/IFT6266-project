import logging

import h5py as h5
import PIL.Image
import numpy as np

import dataset
import mlp

fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
logging.basicConfig(level='DEBUG', format=fmt)
log = logging.getLogger('test_project')

netw = mlp.MLP([(64*64 - 32*32)*3, 1000, 32*32*3], lr=0.0001)

h5dataset = h5.File('mlp_dataset.h5', 'r')

# test figure
xt = h5dataset['val/input'][0]
yt = h5dataset['val/target'][0]

def __gen(batch_size):

    x_train = h5dataset['train/input']
    y_train = h5dataset['train/target']

    while True:
        n = x_train.shape[0]
        for i in range(0, n, batch_size):
            x = x_train[i:i+batch_size]
            y = y_train[i:i+batch_size]
            yield x, y

datagen = __gen(1000)
    
for i in range(10):
    b = netw(xt[np.newaxis, :])
    img = dataset.reconstruct_from_flat(xt, b[0])
    PIL.Image.fromarray(img.astype(np.uint8)).save('test_image_{}.jpg'.format(i))
    log.info('epoch %d', i)
    netw.train_with_generator(datagen, 1, h5dataset['train/input'].shape[0])

netw.save('test_network.h5')
