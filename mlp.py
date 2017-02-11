""" Densely connected multi-layer perceptron """

# pylint: disable=invalid-name, no-member, too-few-public-methods

import logging
import os

import h5py as h5
import numpy as np
import theano
import theano.tensor as T

theano.config.floatX = 'float32'
# uncomment for easier debugging:
# theano.config.optimizer='fast_compile'


log = logging.getLogger(__name__)


class LinearTransformation(object):
    """ Linear transformation of the form X * W + b """

    def __init__(self, shape):
        """ Linear transformation of the form X * W + b

        Arguments:
          shape : tuple (n_in, n_out) defining the dimensions of W
        """

        self.shape = shape
        n_in, n_out = self.shape

        log.debug('Creating LinearTransformation layer with shape n_in=%d, n_out=%d', n_in, n_out)

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(1.0 / n_in),
                high=np.sqrt(1.0 / n_in),
                size=(n_in, n_out),
            ).astype('float32'),
        )

        self.b = theano.shared(
            value=np.zeros(n_out).astype('float32'),
        )

        self.parameters = [self.W, self.b]

    def expression(self, X):
        """ Create a symbolic expression X * W + b """
        return T.dot(X, self.W) + self.b


class MLP(object):
    """ Multilayer perceptron """

    def __init__(self, structure, lr=0.01):

        self.layers = []
        self.parameters = []
        self.lr = lr

        for ni, no in zip(structure[:-1], structure[1:]):
            self.layers.append(LinearTransformation((ni, no)))
            self.parameters += self.layers[-1].parameters

        self.__compile_training_function()
        self.__compile_test_function()

    def train(self, X, Y, n_epochs, batch_size=128):
        """ Train the mlp

        Arguments:
          X: Input data, of shape (N_SAMPLES, N_FEATURES)
          Y: Target data, of shape (N_SAMPLES, N_TARGETS)
          n_epochs: number of passes through whole dataset
          batch_size: number of examples seen before updating parameters
        """
        for epoch in range(n_epochs):
            losses = []
            for i_batch in range(X.shape[0] / batch_size):
                i0 = i_batch * batch_size
                i1 = i0 + batch_size
                losses.append(self.__train_fun(X[i0:i1], Y[i0:i1]))
            log.info('epoch %d: loss=%f', epoch, np.mean(losses))

    def __call__(self, X):
        return self.__test_fun(X)

    def expression(self, X):
        """ Symbolic expression reprensenting the forward propagation """
        tensor = X
        for i, layer in enumerate(self.layers):
            tensor = layer.expression(tensor)
            if i < len(self.layers) - 1:
                tensor = T.nnet.relu(tensor)
        return tensor

    def __compile_training_function(self):
        """Compile a theano function for training

        The resulting function takes 2 minibatches, X and Y, as input
        and outputs the loss. The weights are updated for every
        minibatch.

        Arguments:
          lr : (optional) learning rate for weight updated.
        Returns: a theano function.
        """

        X = T.matrix('X')
        Y = T.matrix('Y')

        loss = T.mean(T.pow(self.expression(X) - Y, 2))

        gparams = [T.grad(loss, p) for p in self.parameters]

        updates = [
            (param, param - self.lr * gparam)
            for param, gparam in zip(self.parameters, gparams)
        ]

        self.__train_fun = theano.function(
            inputs=[X, Y],
            outputs=loss,
            updates=updates,
        )

    def __compile_test_function(self):
        """Compile a theano function for testing

        The resulting function takes 1 minibatche X  as input
        and outputs the prediction.

        Returns: a theano function.
        """

        X = T.matrix('X')

        self.__test_fun = theano.function(
            inputs=[X],
            outputs=self.expression(X)
        )

    def save(self, path):

        uniq = path
        i = 1
        while os.path.exists(uniq):
            uniq = path + '.{}'.format(i)
            i+= 1
            
        savefile = h5.File(path, 'x')

        for i, layer in enumerate(self.layers):
            savefile.create_dataset(
                name='layer{}/W'.format(i),
                data=layer.parameters[0].get_value()
            )
            savefile.create_dataset(
                name='layer{}/b'.format(i),
                data=layer.parameters[1].get_value()
            )

        savefile.create_dataset('n_layers', data=len(self.layers))
        savefile.close()
            
def load(path):
    lfile = h5.File(path, 'r')
    n_layers = lfile['n_layers'].value

    Ws = []
    bs = []
    structure = []
    for i in range(n_layers):
        Ws.append(np.array(lfile['layer{}/W'.format(i)]))
        bs.append(np.array(lfile['layer{}/b'.format(i)]))
        structure.append(bs[-1].shape[0])

    structure = [Ws[0].shape[0]] + structure

    net = MLP(structure)

    for i in range(n_layers):
        net.layers[i].parameters[0].set_value(Ws[i])
        net.layers[i].parameters[1].set_value(bs[i])

    return net
