import cPickle
import h5py as h5

from dataset.words import EncoderDecoder

dset = h5.File('word_dataset.h5')
decoder = cPickle.loads(dset.attrs["encoder_decoder"].tostring())

print '==> decoding a few strings...'
for i in range(25):
    print '  -> {}: {}'.format(i, decoder.decode_sentence(dset['train'][i]))
