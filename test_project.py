import logging

import PIL.Image
import numpy as np

import dataset
import mlp

logging.basicConfig(level='DEBUG')
log = logging.getLogger('test_project')

# test figure
path = '/home/glg/projets/ift6266_projet/IFT6266-project/inpainting/train2014/COCO_train2014_000000149429.jpg'
xt, yt = dataset.get_flattened_example(dataset.load_image(path))


datapath = dataset.retrieve()
datagen = dataset.generator(datapath, batch_size=128)

netw = mlp.MLP([(64*64 - 32*32)*3, 1000, 32*32*3], lr=0.0001)

# preview
for i in range(10):
    b = netw(xt[np.newaxis, :])
    img = dataset.reconstruct_from_flat(xt, b[0])
    PIL.Image.fromarray(img.astype(np.uint8)).save('test_image_{}.jpg'.format(i))
    log.info('epoch %d', i)
    netw.train_with_generator(datagen, 1, dataset.train_set_size(datapath))

netw.save('test_network.h5')
