import logging
import unittest
import theano.tensor as T
import theano
import numpy as np
import os
import h5py as h5
import gzip
import cPickle

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

    def reg_loss(self):
        """ Regularization term to add to loss """
        return 0

class LinearTransformation(Layer):
    """ Linear transformation of the form X * W + b """

    def __init__(self, shape, l2=0.0):
        """ Linear transformation of the form X * W + b

        Arguments:
          shape : tuple (n_in, n_out) defining the dimensions of W
        """

        self.l2 = l2
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

    def reg_loss(self):
        return self.l2 * self.W.norm(2)

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

    def compile(self, lr, momentum):

        self.__train_fun = self.__make_training_function(lr, momentum)
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
        with gzip.open(path, 'wb') as savefile:
            cPickle.dump(self, savefile)

    @staticmethod
    def load(path):
        with gzip.open(path, 'rb') as savefile:
            return cPickle.load(savefile)

    def __make_training_function(self, lr, momentum=0.0):

        X = T.matrix()
        Y = T.matrix()

        self.velocity = []
        for param in self.parameters:
            self.velocity.append(
                theano.shared(
                    np.zeros_like(param.get_value()).astype('float32')
                )
            )

        loss = mse_loss(self.expression(X), Y) + \
               sum([lyr.reg_loss() for lyr in self.layers])

        gparams = [T.grad(loss, param) for param in self.parameters]

        v_updates = [
            (velo, momentum * velo - lr * gparam)
            for velo, param, gparam in zip(self.velocity, self.parameters, gparams)
        ]
        
        p_updates = [
            (param, param + velo)
            for param, velo in zip(self.parameters, self.velocity)
        ]

        return theano.function(
            inputs=[X, Y],
            outputs=loss,
            updates=(v_updates + p_updates)
        )

    def __make_test_function(self):
        X = T.matrix()
        return theano.function(
            inputs=[X],
            outputs=self.expression(X)
        )
  

