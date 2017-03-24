""" Functions for dealing with the dataset """
import glob
import logging
import os
import subprocess
import urllib

import h5py as h5
import numpy as np
import PIL.Image

URL = "http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2"

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


#### function to use to create HDF5 dataset for input

def get_path_list(datadir):
    log.info('Getting good paths from %s', datadir)
    good = []
    paths = glob.glob(datadir + '/*.jpg')
    n_paths = len(paths)
    for i, pth in enumerate(paths):
        if i % 1000 == 0:
            log.debug('%d/%d', i, n_paths)
        img = load_image(pth)
        if len(img.shape) == 3:
            good.append(pth)
    log.info('Found %d good paths out of %d', len(good), n_paths)
    return good

def generate_flattened(datadir):
    for path in get_path_list(datadir):
        img = load_image(path)
        yield get_flattened_example(img)


def create_mlp_dataset(dataset_path, output_path):
    dataset_path = retrieve()

    def __create(dset, h5file):
        
        x_maxshape = (None,) + ((64*64 - 32*32) * 3,)
        y_maxshape = (None,) + (32*32 * 3,)
        
        x_dset = outf.create_dataset(
            name=(dset+'/input'),
            shape=(1, x_maxshape[1]),
            maxshape=x_maxshape,
            compression='lzf',
        )
        y_dset = outf.create_dataset(
            name=(dset+'/target'),
            shape=(1, y_maxshape[1]),
            maxshape=y_maxshape,
            compression='lzf',
        )

        gen = generate_flattened('{}/{}2014'.format(dataset_path, dset))
        x, y = gen.next()
        x_dset[:] = x
        y_dset[:] = y
        count = 1

        for x,y in gen:
            x_dset.resize(count + 1, axis=0)
            y_dset.resize(count + 1, axis=0)
            x_dset[count:] = x
            y_dset[count:] = y
            count += 1
            if count % 100 == 0:
                print count

    with h5.File(output_path, 'w') as outf:
        __create('train', outf)
        __create('val', outf)


def create_conv_dataset(output_path):

    dataset_path = retrieve()

    def __create(dset, h5file):
        
        x_maxshape = (None, 3, 64, 64)
        y_maxshape = (None, 3, 32, 32)
        
        x_dset = outf.create_dataset(
            name=(dset+'/input'),
            shape=((1,) + x_maxshape[1:]),
            maxshape=x_maxshape,
            compression='lzf',
        )
        y_dset = outf.create_dataset(
            name=(dset+'/target'),
            shape=((1,) + y_maxshape[1:]),
            maxshape=y_maxshape,
            compression='lzf',
        )

        gen = generate_unflattened('{}/{}2014'.format(dataset_path, dset))
        x, y = gen.next()
        x_dset[:] = x
        y_dset[:] = y
        count = 1

        for x,y in gen:
            x_dset.resize(count + 1, axis=0)
            y_dset.resize(count + 1, axis=0)
            x_dset[count:] = x
            y_dset[count:] = y
            count += 1
            if count % 100 == 0:
                print count

    with h5.File(output_path, 'w') as outf:
        __create('train', outf)
        __create('val', outf)


def generate_unflattened(datadir):
    for path in get_path_list(datadir):
        img = load_image(path)
        yield get_unflattened_example(img)



#####
        
def retrieve():
    """ Download and/or uncompress the dataset as necessary

    Returns: path to dataset directory
    """

    if 'DATAPATH' in os.environ:
        return os.environ['DATAPATH']
    
    local_path = os.path.basename(URL)
    uncompressed = local_path.replace('.tar.bz2', '')

    if not os.path.exists(uncompressed) and not os.path.exists(local_path):
        log.info('Downloading the dataset')
        urllib.urlretrieve(url=URL, filename=local_path)

    else:
        log.info('Found local copy of dataset')

    if not os.path.exists(uncompressed):
        log.info('Uncompressing the dataset')
        subprocess.check_call(['tar', 'xjf', local_path])
    else:
        log.info('Found local copy of uncompressed dataset')

    return os.path.abspath(uncompressed)


def load_image(path):
    """ load image tensor from path """
    return np.array(PIL.Image.open(path)).astype('float32')


def extract_patch(tensor, flatten):
    """ extract the center 32x32 patch from a 64x64 image """
    patch = tensor[16:48, 16:48]
    return patch.flatten() if flatten else patch


def extract_border(tensor):
    """ extract the border around a  32x32 patch from a 64x64 image """
    border_a = tensor[0:16, 0:48].flatten()
    border_b = tensor[16:64, 0:16].flatten()
    border_c = tensor[48:64, 16:64].flatten()
    border_d = tensor[0:48, 48:64].flatten()
    return np.append(border_a, [border_b, border_c, border_d])

def mask_patch(tensor):
    """ extract the border around a  32x32 patch from a 64x64 image """
    maskd = np.array(tensor, copy=True)
    maskd[16:48, 16:48] = 0
    return maskd

def get_flattened_example(tensor):
    """ get flattened input tuple, suitable for an mlp """
    return extract_border(tensor), extract_patch(tensor, flatten=True)

def get_unflattened_example(tensor):
    """ get flattened input tuple, suitable for an mlp """
    masked = mask_patch(tensor)
    patch = extract_patch(tensor, flatten=True)
    return np.transpose(masked, (2,0,1)), patch

def reconstruct_from_flat(border, patch):
    """ reconstruct image from flattened border and patch """
    tensor = np.zeros((64, 64, 3))
    tensor[16:48, 16:48] = patch.reshape((32, 32, 3))
    bdim = 16 * 48 * 3
    tensor[0:16, 0:48] = border[0:bdim].reshape((16, 48, 3))
    tensor[16:64, 0:16] = border[bdim:bdim*2].reshape((48, 16, 3))
    tensor[48:64, 16:64] = border[bdim*2:bdim*3].reshape((16, 48, 3))
    tensor[0:48, 48:64] = border[bdim*3:bdim*4].reshape((48, 16, 3))
    return tensor

def reconstruct_from_unflat(masked, patch):
    """ reconstruct image from flattened border and patch """
    tensor = np.array(np.transpose(masked, (1, 2, 0)), copy=True)
    tensor[16:48, 16:48] = patch.reshape((32, 32, 3))
    return tensor


def train_set_size(path):
    return len(glob.glob(path + '/train2014/*.jpg'))


def generator(path, batch_size):
    paths = glob.glob(path + '/train2014/*.jpg')
    nsamples = train_set_size(path)
    xbatch = np.zeros((batch_size, (64*64 - 32*32)*3), dtype='float32')
    ybatch = np.zeros((batch_size, 32*32*3), dtype='float32')
    while True:
        ibase = 0
        while ibase < nsamples - batch_size:
            for j in range(0, batch_size):
                while True:
                    img = load_image(paths[ibase + j])
                    ibase += 1
                    if len(img.shape) == 3:
                        break
                x, y = get_flattened_example(img)
                xbatch[j] = x
                ybatch[j] = y
            yield xbatch, ybatch
