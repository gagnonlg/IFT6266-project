import dataset
import pylab
import PIL.Image
import numpy as np

datapath = 'inpainting/val2014'
paths = dataset.get_path_list(datapath)

img = dataset.load_image(paths[0])

PIL.Image.fromarray(img.astype('uint8')).show()

gen = dataset.generate_unflattened(datapath)
masked, patch = gen.next()

PIL.Image.fromarray(np.transpose(masked, (1, 2, 0)).astype('uint8')).show()
PIL.Image.fromarray(patch.reshape(32, 32, 3).astype('uint8')).show()

img_ = dataset.reconstruct_from_unflat(masked, patch)
PIL.Image.fromarray(img_.astype('uint8')).show()

