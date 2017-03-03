import logging
import unittest
import theano.tensor as T
import theano
import numpy as np
import os
import h5py as h5

theano.config.floatX = 'float32'

log = logging.getLogger(__name__)


def mse_loss(xmatrix, ymatrix):
    return T.mean(T.pow(xmatrix - ymatrix, 2).sum(axis=1))

########################################################################
# LAYERS

# API
class Layer(object):
    def shape(self):
        return ()
    
    def expression(self, X):
        """ Return a theano symbolic expression for this layer """
        return X
    
    def parameters(self):
        """ Return the trainable parameters """
        return []

class LinearTransformation(Layer):
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

    def expression(self, X):
        """ Create a symbolic expression X * W + b """
        return T.dot(X, self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Layer):
    def expression(self, X):
        return T.nnet.relu(X)


########################################################################
# Network <=> a collection of layers
class Network(object):

    def __init__(self):
        self.layers = []
        self.parameters = []

    def __call__(self, X):
        return self.__test_fun(X)

    def add(self, layer):
        self.layers.append(layer)
        self.parameters += layer.parameters()

    def compile(self, lr):

        self.__train_fun = self.__make_training_function(lr)
        self.__test_fun = self.__make_test_function()

    def expression(self, X):
        tensor = X
        for i, layer in enumerate(self.layers):
            tensor = layer.expression(tensor)
        return tensor

    def train(self, X, Y, n_epochs, batch_size):
        for epoch in range(n_epochs):
            losses = []
            for i_batch in range(X.shape[0] / batch_size):
                i0 = i_batch * batch_size
                i1 = i0 + batch_size
                losses.append(self.__train_fun(X[i0:i1], Y[i0:i1]))
            log.info('epoch %d: loss=%f', epoch, np.mean(losses))

    def train_with_generator(self, generator, n_epochs, samples_per_epoch):
        for epoch in range(n_epochs):
            seen = 0
            losses = []
            while seen < samples_per_epoch:
                logging.debug('seen: %d of %d', seen, samples_per_epoch)
                xbatch, ybatch = generator.next()
                seen += xbatch.shape[0]
                losses.append(self.__train_fun(xbatch, ybatch))
            log.info('epoch %d: loss=%f', epoch, np.mean(losses))

    def save(self, path):

        uniq = path
        i = 1
        while os.path.exists(uniq):
            uniq = path + '.{}'.format(i)
            i+= 1
            
        savefile = h5.File(uniq, 'x')

        for i, layer in enumerate(self.layers):
            grp = savefile.create_group("layer{}".format(i))
            grp.attrs['type'] = type(layer).__name__
            for j, param in enumerate(layer.parameters()):
                grp.create_dataset(
                    name='param{}'.format(j),
                    data=layer.parameters()[j].get_value()
                )

        savefile.create_dataset('n_layers', data=len(self.layers))
        savefile.close()

    def __make_training_function(self, lr):

        X = T.matrix()
        Y = T.matrix()

        loss = mse_loss(self.expression(X), Y)
        gparams = [T.grad(loss, param) for param in self.parameters]

        updates = [
            (param, param - lr * gparam)
            for param, gparam in zip(self.parameters, gparams)
        ]

        return theano.function(
            inputs=[X, Y],
            outputs=loss,
            updates=updates
        )

    def __make_test_function(self):
        X = T.matrix()
        return theano.function(
            inputs=[X],
            outputs=self.expression(X)
        )
  

