""" Functions for dealing with the dataset """
import logging
import os
import subprocess
import urllib

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
