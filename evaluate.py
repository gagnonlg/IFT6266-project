import argparse
import logging
import re
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import h5py as h5
import scipy.stats

import dataset
import network

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', required=True)
    args.add_argument('--data', required=True)
    args.add_argument('--log')
    args.add_argument('--GAN', action='store_true')
    return args.parse_args()

def montage(borders, centers, outpath):

    imfiles = []
    for i in range(centers.shape[0]):
        img = dataset.reconstruct_from_flat(
            borders[i],
            centers[i]
        ).astype(np.uint8)
        _, tmp = tempfile.mkstemp()
        PIL.Image.fromarray(img).save(tmp, format='jpeg')
        imfiles.append(tmp)

    subprocess.check_call(
        ['montage'] + imfiles + ['-geometry', '+2+2', outpath]
    )
    subprocess.check_call(['rm'] + imfiles)

def loss_curves(log, out):

    loss = []
    v_loss = []
    for line in open(log, 'r').readlines():
        match = re.match('.* loss=(.*), vloss=(.*)', line)
        if match is not None:
            loss.append(float(match.group(1)))
            v_loss.append(float(match.group(2)))

    plt.plot(loss, label='Training')
    plt.plot(v_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(out)
    plt.close()

def entropy(borders, centers, outpath):
    entrs = np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        pp, _ = np.histogram(centers[i], bins=range(2**8), density=True)
        pb, _ = np.histogram(borders[i], bins=range(2**8), density=True)
        entrs[i] = scipy.stats.entropy(pp + 1e-8, pb + 1e-8)

    plt.hist(entrs, histtype='step', bins=50)
    plt.xlabel('KL(center||border)')
    plt.ylabel('Images')
    plt.savefig(outpath)
    plt.close()

def main():

    fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level='INFO', format=fmt)
    log = logging.getLogger('evaluate.py')

    args = get_args()
    log.info('Loading model')
    model = network.Network.load(args.model)
    data = h5.File(args.data, 'r')

    log.info('Generating test images')
    borders = data['val/input'][:1000]
    if args.GAN:
        borders_normed = borders * 2.0 / 255.0 - 1.0
        centers_normed = model(
            np.concatenate(
                [borders_normed, np.random.uniform(size=(1000, 2000)).astype('float32')],
                axis=1
            )
        )[:,borders.shape[1]:]
        centers = (centers_normed + 1.0) * 255.0 / 2.0
    else:
        centers = model(borders)

    log.info('Measuring KL divergence')
    entropy(borders, centers, args.model.replace('.h5', '.entropy.jpg'))

    log.info('Creating montage')
    montage(
        borders=borders[:100],
        centers=centers[:100],
        outpath=args.model.replace('.h5', '.jpg')
    )

    if args.log is not None:
        log.info('Creating loss curves')
        out = args.model.replace('.h5', '') + '.loss.jpg'
        loss_curves(args.log, out)


if __name__ == '__main__':
    main()
