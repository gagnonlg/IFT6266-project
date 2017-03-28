import dataset
import PIL.Image
import numpy as np

import h5py as h5

datapath = 'conv_dataset.h5'
dset = h5.File(datapath)

border = dset['train/input'][0]
patch = dset['train/target'][0]

assert border.shape == (3, 64, 64)
assert patch.shape == (3 * 32 * 32,)

PIL.Image.fromarray(np.transpose(border, (1, 2, 0)).astype('uint8')).show()
PIL.Image.fromarray(patch.reshape((32, 32, 3)).astype('uint8')).show()

img = dataset.reconstruct_from_unflat(border, patch)
PIL.Image.fromarray(img.astype('uint8')).show()
