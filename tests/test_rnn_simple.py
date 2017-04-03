import network
import numpy as np
import theano.tensor as T
import theano
import logging

#theano.config.optimizer='fast_compile'

logging.basicConfig(level='INFO')

np.random.seed(893)


# generate data
def gen_sample():
    mean = [
        [0.0, 0.5],
        [0.5, 0.0]
    ]
    scale = [
        [0.15, 0.15],
        [0.25, 0.25]
    ]

    n = np.random.randint(2,7)

    i = np.random.randint(2)
    
    return np.random.normal(mean[i], scale[i], (n,2)), i

def make_dataset(N):

    dsetX = np.zeros((N, 6, 2))
    dsetY = np.zeros((N, 2))
    dsetM = np.zeros((N, 6))
    for i in range(N):
        smp, cls = gen_sample()
        dsetX[i,:smp.shape[0]] = smp
        dsetY[i,cls] = 1
        dsetM[i,:smp.shape[0]] = 1

    return dsetX.astype('float32'), dsetY.astype('float32'), dsetM.astype('float32')


print '-> generating data'
trainX, trainY, trainM = make_dataset(1000)
validX, validY, _ = make_dataset(1000)
testX, testY, _ = make_dataset(1000)

print '-> building network'
rnn = network.Network()
rnn.add(network.Recurrent(n_feature=2, n_state=5, n_out=1, state_only=True))
rnn.add(network.LinearTransformation((5, 2)))
rnn.add(network.Softmax())
rnn.compile(
    lr=0.001,
    momentum=0.0,
    batch_size=10,
    cache_size=(1000, (6, 2), (2,)),
    vartype=(T.tensor3, T.matrix),
    loss=network.negative_log_likelihood_loss
)

print '-> training network'
rnn.train(trainX, trainY, val_data=(validX, validY), n_epochs=100)

print '-> testing network'
preds = rnn(testX)
good = np.count_nonzero(np.argmax(preds,axis=1) == np.argmax(testY,axis=1))
print 'acc: %f' % (good / 1000.0)
