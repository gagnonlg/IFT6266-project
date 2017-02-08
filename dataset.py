""" Functions for dealing with the dataset """
import logging
import os
import subprocess
import urllib

import numpy as np
import PIL.Image

URL = "http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2"

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def retrieve():
    """ Download and/or uncompress the dataset as necessary

    Returns: path to dataset directory
    """
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
    patch = tensor[16:48, 16:48].flatten()
    return patch.flatten() if flatten else patch


def extract_border(tensor):
    """ extract the border around a  32x32 patch from a 64x64 image """
    border_a = tensor[0:16, 0:48].flatten()
    border_b = tensor[16:64, 0:16].flatten()
    border_c = tensor[48:64, 16:64].flatten()
    border_d = tensor[0:48, 48:64].flatten()
    return np.append(border_a, [border_b, border_c, border_d])


def get_flattened_example(tensor):
    """ get flattened input tuple, suitable for an mlp """
    return extract_border(tensor), extract_patch(tensor, flatten=True)


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
