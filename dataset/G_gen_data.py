import argparse

import h5py as h5
import numpy as np

import dataset
import network


def main():
    args = get_args()
    create_dataset(args.data, args.model, args.output)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', required=True)
    args.add_argument('--data', required=True)
    args.add_argument('--output', required=True)
    return args.parse_args()


def create_dataset(datapath, modelpath, outputpath):

    with h5.File(outputpath, 'w') as outf:
        create_group('train', datapath, modelpath, outf)
        create_group('val', datapath, modelpath, outf)

def create_group(grpname, datapath, modelpath, outf):
    maxshape = (None, 3, 64, 64)
    xdset = outf.create_dataset(
        name=grpname,
        shape=((1,) + maxshape[1:]),
        maxshape=maxshape,
        compression='lzf'
    )

    gen = generate_images('{}/{}2014'.format(datapath, grpname), modelpath)
    x = gen.next()
    xdset[:] = x
    count = 1

    for x in gen:
        xdset.resize(count + 1, axis=0)
        xdset[count] = x
        count += 1
        if count % 100 == 0:
            print count

def generate_images(datapath, modelpath):

    model = network.Network.load(modelpath, test_only=True)

    for path in dataset.get_path_list(datapath):
        img = dataset.load_image(path)

        flat_border = dataset.extract_border(img)
        flat_center = model(flat_border[np.newaxis,:])

        img[16:48, 16:48] = flat_center.reshape((32, 32, 3))

        yield np.transpose(img, (2, 0, 1))

if __name__ == '__main__':
    main()
