import logging
import unittest
import theano.tensor as T
import theano
import numpy as np
import os
import h5py as h5
import gzip
import cPickle
import itertools

theano.config.floatX = 'float32'
theano.config.reoptimize_unpickled_function = True

log = logging.getLogger(__name__)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def mse_loss(xmatrix, ymatrix):
    return T.mean(T.pow(xmatrix - ymatrix, 2).sum(axis=1))

########################################################################
# LAYERS

# API
class Layer(object):

    def expression(self, X):
        """ Return a theano symbolic expression for this layer """
        return X

    def training_expression(self, X):
        return self.expression(X)
    
    def parameters(self):
        """ Return the trainable parameters """
        return []

    def reg_loss(self):
        """ Regularization term to add to loss """
        return 0

    def updates(self):
        return []


class Dropout(Layer):
    def __init__(self, drop_prob, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

        self.keep_prob = 1 - drop_prob

    def training_expression(self, X):
        self.stream = self.rng.uniform(size=X.shape)
        self.mask = T.cast(self.stream < self.keep_prob, 'float32')
        return self.mask * X

    def expression(self, X):
        return X * self.keep_prob
        

class ScaleOffset(Layer):

    def __init__(self, scale=1.0, offset=0.0):

        self.scale = scale
        self.offset = offset

    def expression(self, X):
        return X * self.scale + self.offset

class Clip(Layer):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def expression(self, X):
        return X.clip(self.min, self.max)

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
                low=-np.sqrt(6.0 / (n_in + n_out)),
                high=np.sqrt(6.0 / (n_in + n_out)),
                size=(n_in, n_out),
            ).astype('float32'),
        )

        self.b = theano.shared(
            value=np.full((n_out,), 0.1).astype('float32'),
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


class Sigmoid(Layer):
    def expression(self, X):
        return T.nnet.sigmoid(X)


class BatchNorm(Layer):

    def __init__(self, n_input):
        
        self.gamma = theano.shared(np.float32(1.0))
        self.beta = theano.shared(np.float32(0.0))

        self.online_mean = theano.shared(
            np.zeros(n_input).astype('float32')
        )
        self.online_variance = theano.shared(
            np.ones(n_input).astype('float32')
        )

    def expression(self, X):
        normd = (X - self.online_mean) / T.sqrt(self.online_variance + 0.001)
        return self.gamma * normd + self.beta
        
    def training_expression(self, X):
        self.sample_mean = T.mean(X, axis=0)
        self.sample_variance = T.var(X, axis=0)
        normd = (X - self.sample_mean) / T.sqrt(self.sample_variance + 0.001)
        return self.gamma * normd + self.beta    

    def parameters(self):
        return [self.gamma, self.beta]

    def updates(self):
        mean_upd = self.online_mean * 0.99 + self.sample_mean * 0.01
        var_upd = self.online_variance * 0.99 + self.sample_variance * 0.01
        return [
            (self.online_mean, mean_upd),
            (self.online_variance, var_upd)
        ]

            
    

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

    def compile(self, lr, momentum, batch_size, cache_size):

        self.lr = lr

        self.batch_size = batch_size
        self.cache_size = cache_size

        self.__train_fun = self.__make_training_function(momentum)
        self.__test_fun = self.__make_test_function()
        self.__valid_fun = self.__make_validation_function()
                

    def training_expression(self, X):
        tensor = X
        for i, layer in enumerate(self.layers):
            tensor = layer.training_expression(tensor)
        return tensor

    def expression(self, X):
        tensor = X
        for i, layer in enumerate(self.layers):
            tensor = layer.expression(tensor)
        return tensor


    def train(self,
              X,
              Y,
              val_data,
              n_epochs,
              start_epoch=0):
  
        for epoch in range(start_epoch, start_epoch + n_epochs):

            # First, run the training for the current epoch
            loss = self.__run_training_epoch(X, Y)

            # Bail if the loss is NaN
            if np.isnan(loss):
                log.error('loss is nan, aborting')
                raise RuntimeError('Loss is NaN')

            # Now, compute the validation loss
            vloss = self.__validation_loss(val_data[0], val_data[1])
            log.info(
                'epoch %d: loss=%f, vloss=%f',
                epoch,
                loss,
                vloss,
            )

    def __run_training_epoch(self, X, Y):
        losses = []
        for ibatch in self.__cache_generator(X, Y):
            losses.append(self.__train_fun(ibatch, self.lr))
            if log.isEnabledFor(logging.DEBUG):
                # costly operation...
                bound0 = ibatch * self.batch_size
                bound1 = (ibatch + 1) * self.batch_size
                x = self.X_cache.get_value()[bound0:bound1]
                y = self.Y_cache.get_value()[bound0:bound1]
                log.debug('xbatch: %s, ybatch: %s', str(x.shape), str(y.shape))
        return np.mean(losses)
    
    def __validation_loss(self, Xv, Yv):
        losses = []
        for ibatch in self.__cache_generator(Xv, Yv, batch_size=self.cache_size[0]):
            losses.append(self.__valid_fun(ibatch))
        return np.mean(losses)

    def __cache_generator(self, X, Y, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        for idx in grouper(range(X.shape[0]), self.cache_size[0]):
            idx_ = filter(lambda n: n is not None, idx)
            i0 = idx_[0]
            i1 = idx_[-1]
            self.X_cache.set_value(X[i0:i1])
            self.Y_cache.set_value(Y[i0:i1])
            bs = min(batch_size, len(idx_))
            for ibatch in range(0, len(idx_)/bs):
                yield ibatch

    def save(self, path):
        with gzip.open(path, 'wb') as savefile:
            cPickle.dump(self, savefile)

    @staticmethod
    def load(path):
        with gzip.open(path, 'rb') as savefile:
            return cPickle.load(savefile)

    def __loss(self, X, Y):
        loss = mse_loss(self.training_expression(X), Y)
        for regl in [lyr.reg_loss() for lyr in self.layers]:
            loss = loss + regl
        return loss


    def __make_training_function(self, momentum=0.0):
        nrows_cache = self.cache_size[0]
        ncols_X_cache = self.cache_size[1]
        ncols_Y_cache = self.cache_size[2]
        
        self.X_cache = theano.shared(
            np.zeros((nrows_cache, ncols_X_cache)).astype('float32')
        )

        self.Y_cache = theano.shared(
            np.zeros((nrows_cache, ncols_Y_cache)).astype('float32')
        )

        # placeholders
        X = T.matrix('X')
        Y = T.matrix('Y')
        lr = T.scalar()

        self.velocity = []
        for param in self.parameters:
            self.velocity.append(
                theano.shared(
                    np.zeros_like(param.get_value()).astype('float32')
                )
            )

        loss = self.__loss(X, Y)

        gparams = [T.grad(loss, param) for param in self.parameters]

        v_updates = [
            (velo, momentum * velo - lr * gparam)
            for velo, gparam in zip(self.velocity, gparams)
        ]
        
        p_updates = [
            (param, param + velo)
            for param, velo in zip(self.parameters, self.velocity)
        ]

        updates = v_updates + p_updates
        for upd in [lyr.updates() for lyr in self.layers]:
            updates += upd

        index = T.lscalar()

        bound0 = index * self.batch_size
        bound1 = (index + 1) * self.batch_size
        return theano.function(
            inputs=[index, lr],
            outputs=loss,
            updates=updates,
            givens={
                X: self.X_cache[bound0:bound1],
                Y: self.Y_cache[bound0:bound1],
            },
            allow_input_downcast=True
        )

    def __make_test_function(self):
        X = T.matrix()
        return theano.function(
            inputs=[X],
            outputs=self.expression(X),
            allow_input_downcast=True
        )
  
    def __make_validation_function(self):
        X = T.matrix()
        Y = T.matrix()
        index = T.lscalar()
        bound0 = index * self.batch_size
        bound1 = (index + 1) * self.batch_size
        return theano.function(
            inputs=[index],
            outputs=self.__loss(X, Y),
            givens={
                X: self.X_cache[bound0:bound1],
                Y: self.Y_cache[bound0:bound1]
            },
            allow_input_downcast=True
        )
            
