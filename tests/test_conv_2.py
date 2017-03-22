import theano.tensor as T
import network
import gzip
import cPickle as pickle
import numpy as np
import logging
import os

logging.basicConfig(level='DEBUG')

def onehot_mnist(x):
    o = np.zeros((x.shape[0], 10))
    o[np.arange(x.shape[0]), x] = 1
    return o


# data
print "-> Loading data..."
path = os.getenv('MNIST')
if path is None:
    path = 'tests/mnist.pkl.gz'
with gzip.open(path, 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

    tr_x = train_set[0].astype('float32')
    tr_y = onehot_mnist(train_set[1]).astype('float32')
    va_x = valid_set[0].astype('float32')
    va_y = onehot_mnist(valid_set[1]).astype('float32')
    te_x = test_set[0].astype('float32')
    te_y = onehot_mnist(test_set[1]).astype('float32')

    train_set = (tr_x, tr_y)
    valid_set = (va_x, va_y)
    test_set = (te_x, te_y)

#exit()


print '-> Building network...'
conv = network.Network()

# convolve into 20 maps of size
# input shape + filter shape - 1 = 28 + 5 - 1 = 32
# subsample by 2 => 20 16x16 maps
conv.add(network.Convolution(20, 1, 5, 5)) 
conv.add(network.MaxPool((2,2))) # output (batch, 20, 16, 16)
conv.add(network.ReLU())


# convolve into 50 maps of size
# input shape + filter shape - 1 = 16 + 5 - 1 = 20
# subsample by 2 => 50 10x10 maps
conv.add(network.Convolution(50, 20, 5, 5))
conv.add(network.MaxPool((2,2)))
conv.add(network.ReLU())

# flatten into (batch_size, 50 * 10 * 10) = (batch_size, 5000)
conv.add(network.Flatten())

# # Fully connected layers
conv.add(network.LinearTransformation((5000, 500)))
conv.add(network.ReLU())
conv.add(network.LinearTransformation((500, 10)))
conv.add(network.Softmax())

conv.compile(
    lr=0.01,
    momentum=0.0,
    batch_size=256,
    cache_size=(256*5, (1, 28, 28), (10,)),
    vartype=(T.tensor4, T.matrix)
)

print '-> Training network...'

def test():
    xtest = test_set[0].reshape((test_set[0].shape[0],1,28,28))
    test_out = np.argmax(conv(xtest), axis=1)
    good = np.count_nonzero(test_out == np.argmax(test_set[1], axis=1))
    acc = float(good) / test_set[0].shape[0]
    print 'accuracy: {}'.format(acc)

test()

for i in range(100):
    
    conv.train(
        X=train_set[0].reshape((train_set[0].shape[0], 1, 28, 28)),
        Y=train_set[1],
        val_data=(valid_set[0].reshape((valid_set[0].shape[0], 1, 28, 28)), valid_set[1]),
        n_epochs=1
    )

    test()
