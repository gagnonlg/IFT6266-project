import network
import numpy as np
import theano.tensor as T
import theano
import logging
import h5py as h5
import cPickle
import gzip

#theano.config.optimizer='None'

logging.basicConfig(level='INFO')

np.random.seed(893)


# generate data
print '-> loading data'
dset = h5.File('chars_dataset.h5', 'r')
trainX = dset['train']
valX = dset['val']

print '-> building network'

class CustomLayer(network.Layer):

    def expression(self, X):

        def output_step(X):
            charvect = X[:, :-1] # across all minibatch, take only char vector
            stcend = X[:, -1]  # across all minibatch, take the sentence end prob
            retval = T.concatenate([T.nnet.softmax(charvect), T.nnet.sigmoid(stcend).dimshuffle(0,'x')], axis=1)
            return retval

        results, _ = theano.scan(
            fn=output_step,
            outputs_info=None,
            sequences=X
        )

        return results

def custom_loss(x, y):

    def loss_step(X,Y):
        x_charvect = X[:, :-1] # across all minibatch, take only char vector
        x_stcend = X[:, -1]  # across all minibatch, take the sentence end prob

        y_charvect = Y[:, :-1] # across all minibatch, take only char vector
        y_stcend = Y[:, -1]  # across all minibatch, take the sentence end prob

        loss_charvect = T.nnet.categorical_crossentropy(x_charvect, y_charvect)
        loss_stcend = T.nnet.binary_crossentropy(x_stcend, y_stcend)
        
        return loss_charvect + loss_stcend
    
    losses, _ = theano.scan(
        fn=loss_step,
        outputs_info=None,
        sequences=[x, y.dimshuffle(1, 0, 2)]
    )

    return T.mean(T.sum(losses, axis=0))
        


rnn = network.Network()
rnn.add(network.LSTM(n_feature=trainX.shape[2], n_state=100, last_state_only=True))
rnn.add(
    network.LSTM(
        n_feature=100,
        n_state=trainX.shape[2],
        last_state_only=False,
        const_input=True,
        n_step=trainX.shape[1]
    )
)
rnn.add(CustomLayer())
rnn.compile(
    lr=0.01,
    momentum=0.5,
    batch_size=100,
    cache_size=(1000, trainX.shape[1:], trainX.shape[1:]),
    vartype=(T.tensor3, T.tensor3),
    loss=custom_loss
)

chardict = cPickle.loads(dset.attrs["chardict"].tostring())
chardict = {v: k for k, v in chardict.iteritems()}

def test(epoch):
    preds = rnn(valX[:10])
    with gzip.open('val10_{}.gz'.format(epoch), 'w') as testf:
        for i in range(preds.shape[0]):
            stc = ''
            for j in range(preds.shape[1]):
                stc += chardict[np.argmax(preds[i,j,:-1])]
                if preds[i,j,-1] > 0.5:
                    break
            print stc
            testf.write(stc + '\n')

test(-1)
    
print '-> training network'
for i in range(100):
    rnn.train(
        trainX,
        trainX,
        val_data=(valX[:10], valX[:10]),
        n_epochs=1,
        start_epoch=i
    )
    test(i)


