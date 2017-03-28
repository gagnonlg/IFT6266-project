import collections 
import cPickle

import h5py as h5
import numpy as np

import dataset

def cleanup(word):
    remove = ['.', ',' ,';', ':', '\'', '\"']
    for c in remove:
        word = word.replace(c, '')
    return word.lower()
    
def create_word_list(sentences):
    return cleanup(' '.join(sentences)).split()

class EncoderDecoder(object):
    def __init__(self, all_words, n_min=100):
        wdict = collections.defaultdict(int)
        for word in all_words:
            wdict[word] += 1
        self.words_100 = [w for w,n in wdict.iteritems() if n >= n_min]
        self.n_words = len(self.words_100)
    
    def encode(self, word):
        vect = np.zeros(self.n_words + 1, dtype='float32')
        try:
            idx = self.words_100.index(word)
        except ValueError:
            idx = self.n_words
        vect[idx] = 1
        return vect
        
    def decode(self, vect):
        idx = np.argmax(vect)
        if idx == self.n_words:
            return '<UNKNOWN>'
        else:
            return self.words_100[idx]

    def encode_sentence(self, sentence, nwords_max):
        encoded = [self.encode(w) for w in cleanup(sentence).split()]
        vect_size = encoded[0].shape[0]
        matrix = np.zeros((nwords_max, vect_size))
        for i, w in enumerate(encoded):
            matrix[i] = w
        return matrix

    def decode_sentence(self, matrix):
        sentence = []
        for i in range(matrix.shape[0]):
            if np.count_nonzero(matrix[i]) == 0:
                break
            sentence.append(self.decode(matrix[i]))
        return ' '.join(sentence)

    def gen_encoded(self, sentences):
        nwords_max = np.max([len(s.split()) for s in sentences])
        for i in range(len(sentences)):
            yield self.encode_sentence(sentences[i], nwords_max)

def create_word_dataset(output_path, dataset_path=None):
    if dataset_path is None:
        dataset_path = dataset.retrieve()
    
    _sentences = cPickle.load(
        open(
            dataset_path + '/dict_key_imgID_value_caps_train_and_valid.pkl'
        )
    )

    h5file = h5.File(output_path, 'w')

    enc_dec = None
    
    for dset in ['train', 'val']:

        sentences = []
        for stc in [v for k,v in _sentences.iteritems() if dset in k]:
            sentences.append(np.random.choice(stc))

        if enc_dec is None:
            enc_dec = EncoderDecoder(create_word_list(sentences))

        gen = enc_dec.gen_encoded(sentences)
        pilot = gen.next()

        maxshape = (None,) + pilot.shape

        dset = h5file.create_dataset(
            name=dset,
            shape=((1,) + maxshape[1:]),
            maxshape=maxshape,
            compression='lzf'
        )
        
        dset[:] = pilot
        count = 1

        for x in gen:
            dset.resize(count + 1, axis=0)
            dset[count:] = x
            count += 1
            if count % 100 == 0:
                print count

    h5file.attrs["encoder_decoder"] = np.void(cPickle.dumps(enc_dec))

    h5file.close()


if __name__ == '__main__':
    create_word_dataset('word_dataset.h5')
