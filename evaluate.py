import argparse
import logging
import re
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import h5py as h5

import dataset
import network

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', required=True)
    args.add_argument('--data', required=True)
    args.add_argument('--log')
    return args.parse_args()

def montage(model, data, outpath):

    imfiles = []
    centers = model(data)
    for i in range(centers.shape[0]):
        img = dataset.reconstruct_from_flat(
            data[i],
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
            
def main():
    
    fmt = '[%(asctime)s] %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level='INFO', format=fmt)
    log = logging.getLogger('evaluate.py')

    args = get_args()
    log.info('Loading model')
    model = network.Network.load(args.model)
    data = h5.File(args.data, 'r')

    log.info('Creating montage')
    montage(
        model=model,
        data=data['val/input'][:100],
        outpath=args.model.replace('.h5', '.jpg')
    )

    if args.log is not None:
        log.info('Creating loss curves')
        out = args.model.replace('.h5', '') + '.loss.jpg'
        loss_curves(args.log, out)
    

if __name__ == '__main__':
    main()
