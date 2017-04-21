import logging
import unittest
import theano.tensor as T
import theano
import theano.tensor.signal.pool
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

def negative_log_likelihood_loss(x, y):
    return -T.mean(T.log(x)[T.arange(y.shape[0]), T.argmax(y, axis=1)])

def cross_entropy_vector_loss(x, y):
    f = 1.0 / 255
    return T.mean(T.nnet.binary_crossentropy(f*x, f*y).sum(axis=1))

########################################################################
# LAYERS

# API
class Layer(object):

    def expression(self, X):
        """ Return a theano smbolic expression for this layer """
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

    def save(self, h5grp):
        pass

    @classmethod
    def load(cls, h5dset):
        return cls()


def bound(fan_in, fan_out):
    return 1.0 / np.sqrt(fan_in + fan_out)

class LSTM(Layer):

    def __init__(self, n_feature, n_state, last_state_only=False, const_input=False, n_step=None):

        U_bound = bound(n_feature, n_state)
        U_shape = (n_feature, n_state)
        self.U_i = theano.shared(
            np.random.uniform(
                low=-U_bound,
                high=U_bound,
                size=U_shape
            ).astype('float32')
        )
        self.U_f = theano.shared(
            np.random.uniform(
                low=-U_bound,
                high=U_bound,
                size=U_shape
            ).astype('float32')
        )
        self.U_o = theano.shared(
            np.random.uniform(
                low=-U_bound,
                high=U_bound,
                size=U_shape
            ).astype('float32')
        )
        self.U_a = theano.shared(
            np.random.uniform(
                low=-U_bound,
                high=U_bound,
                size=U_shape
            ).astype('float32')
        )
        W_bound = bound(n_feature, n_state)
        W_shape = (n_state, n_state)
        self.W_i = theano.shared(
            np.random.uniform(
                low=-W_bound,
                high=W_bound,
                size=W_shape
            ).astype('float32')
        )
        self.W_f = theano.shared(
            np.random.uniform(
                low=-W_bound,
                high=W_bound,
                size=W_shape
            ).astype('float32')
        )
        self.W_o = theano.shared(
            np.random.uniform(
                low=-W_bound,
                high=W_bound,
                size=W_shape
            ).astype('float32')
        )
        self.W_a = theano.shared(
            np.random.uniform(
                low=-W_bound,
                high=W_bound,
                size=W_shape
            ).astype('float32')
        )

        self.b_i = theano.shared(
            np.zeros(n_state).astype('float32')
        )
        self.b_f = theano.shared(
            np.zeros(n_state).astype('float32')
        )
        self.b_o = theano.shared(
            np.zeros(n_state).astype('float32')
        )
        self.b_a = theano.shared(
            np.zeros(n_state).astype('float32')
        )

        self.n_feature = n_feature
        self.n_state = n_state
        self.last_state_only = last_state_only
        self.const_input = const_input
        self.n_step = n_step

    def expression(self, X):

        def state_step(X, H):

            input_gate = T.nnet.sigmoid(
                self.b_i + T.dot(X, self.U_i) + T.dot(H, self.W_i)
            )

            forget_gate = T.nnet.sigmoid(
                self.b_f + T.dot(X, self.U_f) + T.dot(H, self.W_f)
            )

            output_gate = T.nnet.sigmoid(
                self.b_o + T.dot(X, self.U_o) + T.dot(H, self.W_o)
            )

            activation = T.nnet.sigmoid(
                self.b_a + T.dot(X, self.U_a) + T.dot(H, self.W_a)
            )

            state = activation * input_gate + H * forget_gate

            return T.tanh(state) * output_gate

        def state_step_const(H):
            input_gate = T.nnet.sigmoid(
                self.b_i + T.dot(X, self.U_i) + T.dot(H, self.W_i)
            )

            forget_gate = T.nnet.sigmoid(
                self.b_f + T.dot(X, self.U_f) + T.dot(H, self.W_f)
            )

            output_gate = T.nnet.sigmoid(
                self.b_o + T.dot(X, self.U_o) + T.dot(H, self.W_o)
            )

            activation = T.nnet.sigmoid(
                self.b_a + T.dot(X, self.U_a) + T.dot(H, self.W_a)
            )

            state = activation * input_gate + H * forget_gate

            return T.tanh(state) * output_gate


        initial_state = T.zeros((X.shape[0], self.n_state))

        states, _ = theano.scan(
            fn=(state_step_const if self.const_input else state_step),
            outputs_info=initial_state,
            sequences=(None if self.const_input else X.dimshuffle(1, 0, 2)),
            n_steps=self.n_step
        )

        return states[-1] if self.last_state_only else states


    def parameters(self):
        return [
            self.U_i,
            self.U_f,
            self.U_o,
            self.U_a,
            self.W_i,
            self.W_f,
            self.W_o,
            self.W_a,
            self.b_i,
            self.b_f,
            self.b_o,
            self.b_a,
        ]

    def save(self, h5grp):

        n_step = -1 if self.n_step is None else n_step
        
        h5grp.create_dataset('n_feature', data=self.n_feature)
        h5grp.create_dataset('n_state', data=self.n_state)
        h5grp.create_dataset('last_state_only', data=self.last_state_only)
        h5grp.create_dataset('const_input', data=self.const_input)
        h5grp.create_dataset('n_step', data=n_step)
        h5grp.create_dataset('U_i', data=self.U_i.get_value())
        h5grp.create_dataset('U_f', data=self.U_f.get_value())
        h5grp.create_dataset('U_o', data=self.U_o.get_value())
        h5grp.create_dataset('U_a', data=self.U_a.get_value())
        h5grp.create_dataset('W_i', data=self.W_i.get_value())
        h5grp.create_dataset('W_f', data=self.W_f.get_value())
        h5grp.create_dataset('W_o', data=self.W_o.get_value())
        h5grp.create_dataset('W_a', data=self.W_a.get_value())
        h5grp.create_dataset('b_i', data=self.b_i.get_value())
        h5grp.create_dataset('b_f', data=self.b_f.get_value())
        h5grp.create_dataset('b_o', data=self.b_o.get_value())
        h5grp.create_dataset('b_a', data=self.b_a.get_value())

    @staticmethod
    def load(h5grp):
        n_feature = h5grp['n_feature'].value
        n_state = h5grp['n_state'].value
        last_state_only = h5grp['last_state_only'].value
        const_input = h5grp['const_input'].value
        n_step = h5grp['n_step'].value

        layer = LSTM(
            n_feature=n_feature,
            n_state=n_state,
            last_state_only=last_state_only,
            const_input=const_input,
            n_step=n_step if n_step != -1 else None,
        )

        layer.U_i.set_value(h5grp['U_i'].value)
        layer.U_f.set_value(h5grp['U_f'].value)
        layer.U_o.set_value(h5grp['U_o'].value)
        layer.U_a.set_value(h5grp['U_a'].value)
        layer.W_i.set_value(h5grp['W_i'].value)
        layer.W_f.set_value(h5grp['W_f'].value)
        layer.W_o.set_value(h5grp['W_o'].value)
        layer.W_a.set_value(h5grp['W_a'].value)
        layer.b_i.set_value(h5grp['b_i'].value)
        layer.b_f.set_value(h5grp['b_f'].value)
        layer.b_o.set_value(h5grp['b_o'].value)
        layer.b_a.set_value(h5grp['b_a'].value)

        return layer
        

class Recurrent(Layer):

    def __init__(self, n_feature, n_state, n_out, state_only=False, last_output_only=False):

        self.U = theano.shared(
            np.random.uniform(
                low=-bound(n_feature, n_state),
                high=bound(n_feature, n_state),
                size=(n_feature, n_state),
            ).astype('float32'),
            name='U'
        )
        self.W = theano.shared(
            np.random.uniform(
                low=-bound(n_state, n_state),
                high=bound(n_state, n_state),
                size=(n_state, n_state),
            ).astype('float32'),
            name='W',
        )
        self.V = theano.shared(
            np.random.uniform(
                low=-bound(n_state, n_out),
                high=bound(n_state, n_out),
                size=(n_state, n_out),
            ).astype('float32')
        )
        self.b = theano.shared(
            np.zeros(n_state).astype('float32')
        )
        self.c = theano.shared(
            np.zeros(n_out).astype('float32')
        )

        self.n_feature = n_feature
        self.n_state = n_state
        self.n_out = n_out

        self.state_only = state_only
        self.last_output_only = last_output_only

    def expression(self, X):

        initial_state = T.zeros((X.shape[0], self.n_state))

        def state_step(X, H, U, b, W):
            return T.tanh(b + T.dot(X, U) + T.dot(H, W))

        states, _ = theano.scan(
            fn=state_step,
            outputs_info=initial_state,
            sequences=X.dimshuffle(1, 0, 2),
            non_sequences=[self.U, self.b, self.W]
        )

        if self.state_only:
            return states[-1]

        def pred_step(H, V, c):
            return T.nnet.sigmoid(c + T.dot(H, V))

        preds, _ = theano.scan(
            fn=pred_step,
            outputs_info=None,
            sequences=states,
            non_sequences=[self.V, self.c]
        )

        return preds[-1] if self.last_output_only else preds

    def parameters(self):
        params = [self.U, self.b, self.W]
        return params if self.state_only else params + [self.V, self.c]

    def save(self, h5grp):
        h5grp.create_dataset('n_feature', data=self.n_feature)
        h5grp.create_dataset('n_state', data=self.n_state)
        h5grp.create_dataset('n_out', data=self.n_out)
        h5grp.create_dataset('state_only', data=self.state_only)
        h5grp.create_dataset('last_output_only', data=self.last_output_only)
        h5grp.create_dataset('U', data=self.U.get_value())
        h5grp.create_dataset('W', data=self.W.get_value())
        h5grp.create_dataset('V', data=self.V.get_value())
        h5grp.create_dataset('b', data=self.b.get_value())
        h5grp.create_dataset('c', data=self.c.get_value())

    @staticmethod
    def load(h5grp):
        layer = Recurrent(
            n_feature=h5grp['n_feature'].value,
            n_state=h5grp['n_state'].value,
            n_out=h5grp['n_out'].value,
            state_only=h5grp['state_only'].value,
            last_output_only=h5grp['last_output_only'].value,
        )
        layer.U.set_value(h5grp['U'].value)
        layer.W.set_value(h5grp['W'].value)
        layer.V.set_value(h5grp['V'].value)
        layer.b.set_value(h5grp['b'].value)
        layer.c.set_value(h5grp['c'].value)

        return layer

class Flatten(Layer):

    def expression(self, X):
        return T.flatten(X, outdim=2)

class Convolution(Layer):

    def __init__(self,
                 n_feature_maps,
                 n_input_channels,
                 height,
                 width,
                 l2=0.0,
                 strides=(1,1),
                 border_mode='full'):
        self.filter_shape = (n_feature_maps, n_input_channels, height, width)
        self.border_mode = border_mode
        bound = n_input_channels * height * width
        self.kernel = theano.shared(
            np.random.uniform(
                low=-bound,
                high=bound,
                size=self.filter_shape
            ).astype('float32')
        )

        # ne bias per output feature map
        self.b = theano.shared(
            np.full((n_feature_maps,), 0.1).astype('float32')
        )


        self.l2 = l2
        self.strides = strides

    def expression(self, X):
        # expected shape of input:
        # (batch, channel, height, width)
        return T.nnet.conv2d(
            X,
            self.kernel,
            filter_shape=self.filter_shape,
            border_mode=self.border_mode,
            subsample=self.strides
        ) + self.b.dimshuffle('x', 0, 'x', 'x')

    def parameters(self):
        return [self.kernel, self.b]

    def reg_loss(self):
        return self.l2 * self.kernel.norm(2)

    def save(self, h5grp):
        h5grp.create_dataset('filter_shape', data=self.filter_shape)
        h5grp.create_dataset('border_mode', data=self.border_mode)
        h5grp.create_dataset('kernel', data=self.kernel.get_value())
        h5grp.create_dataset('b', data=self.b.get_value())
        h5grp.create_dataset('l2', data=self.l2)
        h5grp.create_dataset('strides', data=self.strides)

    @staticmethod
    def load(h5grp):
        shape = h5grp['filter_shape'].value
        l2 = h5grp['l2'].value
        strides = h5grp['strides'].value
        border_mode = h5grp['border_mode'].value

        layer = Convolution(
            n_feature_maps=shape[0],
            n_input_channels=shape[1],
            height=shape[2],
            width=shape[3],
            l2=l2,
            strides=strides,
            border_mode=border_mode
        )

        layer.kernel.set_value(h5grp['kernel'].value)
        layer.b.set_value(h5grp['b'].value)

        return layer

class MaxPool(Layer):

    def __init__(self, factors, ignore_border=False):
        self.ignore_border = ignore_border
        self.poolsize = factors

    def expression(self, X):
        return theano.tensor.signal.pool.pool_2d(
            input=X,
            ds=self.poolsize,
            ignore_border=self.ignore_border,
        )

    def save(self, h5grp):
        h5grp.create_dataset('ignore_border', data=self.ignore_border)
        h5grp.create_dataset('poolsize', data=self.poolsize)

    @staticmethod
    def load(h5grp):
        return MaxPool(
            factors=tuple(h5grp['poolsize'].value),
            ignore_border=h5grp['ignore_border'].value
        )

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

    def save(self, h5grp):
        h5grp.create_dataset('keep_prob', data=self.keep_prob)

    @staticmethod
    def load(h5grp):
        return Dropout(drop_prob=(1 - h5grp['keep_prob'].value))


class ScaleOffset(Layer):

    def __init__(self, scale=1.0, offset=0.0):

        self.scale = np.float32(scale)
        self.offset = np.float32(offset)

    def expression(self, X):
        return X * self.scale + self.offset

    def save(self, h5grp):
        h5grp.create_dataset('scale', data=self.scale)
        h5grp.create_dataset('offset', data=self.offset)

    @staticmethod
    def load(h5grp):
        return ScaleOffset(
            scale=h5grp['scale'].value,
            offset=h5grp['offset'].value
        )

class Clip(Layer):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def expression(self, X):
        return X.clip(self.min, self.max)

    def save(self, h5grp):
        h5grp.create_dataset('min', data=self.min)
        h5grp.create_dataset('max', data=self.max)

    @staticmethod
    def load(h5grp):
        return Clip(
            min=h5grp['min'].value,
            max=h5grp['max'].value
        )

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

    def save(self, h5grp):
        h5grp.create_dataset('l2', data=self.l2)
        h5grp.create_dataset('shape', data=self.shape)
        h5grp.create_dataset('W', data=self.W.get_value())
        h5grp.create_dataset('b', data=self.b.get_value())

    @staticmethod
    def load(h5grp):
        layer = LinearTransformation(h5grp['shape'].value)
        layer.l2 = np.float32(h5grp['l2'].value)
        layer.W.set_value(h5grp['W'].value)
        layer.b.set_value(h5grp['b'].value)
        return layer
        

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

class Tanh(Layer):
    def expression(self, X):
        return T.tanh(X)

class Sigmoid(Layer):
    def expression(self, X):
        return T.nnet.sigmoid(X)

class Softmax(Layer):
    def expression(self, X):
        return T.nnet.softmax(X)


class BatchNorm(Layer):

    def __init__(self, n_input):

        self.n_input = n_input

        self.gamma = theano.shared(np.float32(1.0))
        self.beta = theano.shared(np.float32(0.0))

        self.online_mean = theano.shared(
            np.zeros(n_input).astype('float32')
        )
        self.online_variance = theano.shared(
            np.ones(n_input).astype('float32')
        )

    def save(self, h5grp):
        h5grp.create_dataset('n_input', data=self.n_input)
        h5grp.create_dataset('gamma', data=self.gamma.get_value())
        h5grp.create_dataset('beta', data=self.beta.get_value())
        h5grp.create_dataset('online_mean', data=self.online_mean.get_value())
        h5grp.create_dataset('online_variance', data=self.online_variance.get_value())

    @staticmethod
    def load(h5grp):
        layer = BatchNorm(h5grp['n_input'].value)
        layer.gamma.set_value(h5grp['gamma'].value)
        layer.beta.set_value(h5grp['beta'].value)
        layer.online_mean.set_value(h5grp['online_mean'].value)
        layer.online_variance.set_value(h5grp['online_variance'].value)
        return layer
        
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

    def __init__(self, is_GAN_discriminator=False, copy_input=None):
        self.layers = []
        self.parameters = []
        self.best_vloss = np.inf
        self.is_GAN_discriminator = is_GAN_discriminator
        self.copy_input = copy_input
        

    def __call__(self, X):
        return self.__test_fun(X)

    def add(self, layer):
        self.layers.append(layer)
        self.parameters += layer.parameters()

    def compile(self,
                batch_size,
                cache_size,
                lr=0.001,
                momentum=0.0,
                vartype=(T.matrix, T.matrix),
                loss=mse_loss,
                use_ADAM=False):

        self.use_ADAM = use_ADAM
        
        self.loss = loss

        self.vartypeX = vartype[0]
        self.vartypeY = vartype[1]

        self.lr = lr
        self.momentum = momentum

        self.batch_size = batch_size
        self.cache_size = cache_size

        self.__train_fun = self.__make_training_function(momentum, use_ADAM)
        self.__test_fun = self.__make_test_function()
        self.__valid_fun = self.__make_validation_function()


    def __maybe_copy_input(self, input, output):
        if self.copy_input is None:
            return output
        else:
            return T.concatenate(
                [input[:,self.copy_input[0]:self.copy_input[1]], output],
                axis=1
            )

    def training_expression(self, X):
        tensor = X
        for i, layer in enumerate(self.layers):
            tensor = layer.training_expression(tensor)
        return self.__maybe_copy_input(X, tensor)

    def expression(self, X):
        tensor = X
        for i, layer in enumerate(self.layers):
            tensor = layer.expression(tensor)
        return self.__maybe_copy_input(X, tensor)


    def train(self,
              X,
              Y,
              val_data,
              n_epochs,
              start_epoch=0,
              savepath=None):

        for epoch in range(start_epoch, start_epoch + n_epochs):

            # First, run the training for the current epoch
            loss = self.__run_training_epoch(X, Y, epoch)

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

            if (savepath is not None) and (vloss < self.best_vloss):
                log.info(
                    'epoch %d: validation loss improved, saving model (%s)',
                    epoch,
                    savepath
                )
                self.best_vloss = vloss
                self.save(savepath)

    def __run_training_epoch(self, X, Y, epoch):
        losses = []
        for ibatch in self.__cache_generator(X, Y):
            if self.use_ADAM:
                losses.append(self.__train_fun(ibatch, self.lr, epoch))
            else:
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
        with h5.File(path, 'w') as savefile:
            for i, lyr in enumerate(self.layers):
                key = str(i) + ':' + lyr.__class__.__name__
                grp = savefile.create_group(key)
                lyr.save(grp)

            grp = savefile.create_group('Network')
            grp.attrs['loss'] = np.void(cPickle.dumps(self.loss))
            grp.attrs['vartype'] = np.void(cPickle.dumps((self.vartypeX, self.vartypeY)))
            grp.attrs['cache_size'] = np.void(cPickle.dumps(self.cache_size))
            grp.attrs['copy_input'] = np.void(cPickle.dumps(self.copy_input))
            grp.attrs['is_GAN_discriminator'] = np.void(cPickle.dumps(self.is_GAN_discriminator))

            grp.create_dataset('use_ADAM', data=1.0 if self.use_ADAM else 0.0)
            grp.create_dataset('lr', data=self.lr)
            grp.create_dataset('momentum', data=self.momentum)
            grp.create_dataset('batch_size', data=self.batch_size)

    @staticmethod
    def load(path):

        with h5.File(path, 'r') as savefile:

            grp = savefile['Network']

            netw = Network(
                is_GAN_discriminator=get_pickled_attr(grp, 'is_GAN_discriminator', False),
                copy_input=get_pickled_attr(grp, 'copy_input', None)
            )

            ikeys = []
            for key in savefile.keys():
                if key == 'Network':
                    continue
                fields = key.split(':')
                ikeys.append((int(fields[0]), fields[1], key))
            
            for _, type, key in sorted(ikeys):
                log.debug('loading layer type: %s', type)
                netw.add(globals()[type].load(savefile[key]))

            if 'use_ADAM' in grp:
                use_ADAM = grp['use_ADAM'].value == 1,
            else:
                use_ADAM = False
                
            netw.compile(
                lr=np.float32(grp['lr'].value),
                momentum=np.float32(grp['momentum'].value),
                batch_size=grp['batch_size'].value,
                cache_size=cPickle.loads(grp.attrs['cache_size'].tostring()),
                vartype=cPickle.loads(grp.attrs['vartype'].tostring()),
                loss=cPickle.loads(grp.attrs['loss'].tostring()),
                use_ADAM=use_ADAM
            )


        return netw

    def __loss(self, X, Y):
        if self.is_GAN_discriminator:
            loss = self.loss(
                self.training_expression(X),
                self.training_expression(Y)
            )
        else:
            loss = self.loss(self.training_expression(X), Y)
        for regl in [lyr.reg_loss() for lyr in self.layers]:
            loss = loss + regl
        return loss


    def __make_training_function(self, momentum=0.0, adam=False):
        nrows_cache = self.cache_size[0]
        X_cache_size = self.cache_size[1]
        Y_cache_size = self.cache_size[2]

        if not type(X_cache_size) == tuple:
            X_cache_size = (X_cache_size,)
        if not type(Y_cache_size) == tuple:
            Y_cache_size = (Y_cache_size,)

        self.X_cache = theano.shared(
            np.zeros((nrows_cache,) + X_cache_size).astype('float32')
        )

        self.Y_cache = theano.shared(
            np.zeros((nrows_cache,) + Y_cache_size).astype('float32')
        )

        # ADAM constants
        delta = 10e-8
        rho1 = 0.9
        rho2 = 0.999

        # placeholders
        X = self.vartypeX('X')
        Y = self.vartypeY('Y')
        lr = T.scalar()
        t = T.scalar() # epoch

        # will be used if ADAM
        self.moment_1 = []
        self.moment_2 = []
        for param in self.parameters:
            self.moment_1.append(
                theano.shared(
                    np.zeros_like(param.get_value()).astype('float32')
                )
            )
            self.moment_2.append(
                theano.shared(
                    np.zeros_like(param.get_value()).astype('float32')
                )
            )


        # will be used if not ADAM
        self.velocity = []
        for param in self.parameters:
            self.velocity.append(
                theano.shared(
                    np.zeros_like(param.get_value()).astype('float32')
                )
            )

        loss = self.__loss(X, Y)

        gparams = [T.grad(loss, param) for param in self.parameters]


        if adam:
            
            f1 = 1.0 / (1 - T.pow(rho1, t + 1))
            m_1_updates = [
                (m, rho1 * m + (1 - rho1) * g)
                for m, g in zip(self.moment_1, gparams)
            ]

            f2 = 1.0 / (1 - T.pow(rho2, t + 1))
            m_2_updates = [
                (m, rho2 * m + (1 - rho2) * g * g)
                for m, g in zip(self.moment_2, gparams)
            ]

            p_updates = [
                (param, param - lr * (f1 * s) / (T.sqrt(f2*r) + delta))
                for param, s, r in
                zip(self.parameters, self.moment_1, self.moment_2)
            ]

            updates = m_1_updates + m_2_updates + p_updates
            
        else:

            # will be used if not ADAM
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
        if adam:
            inputs = [index, lr, t]
        else:
            inputs = [index, lr]

        bound0 = index * self.batch_size
        bound1 = (index + 1) * self.batch_size
        return theano.function(
            inputs=inputs,
            outputs=loss,
            updates=updates,
            givens={
                X: self.X_cache[bound0:bound1],
                Y: self.Y_cache[bound0:bound1],
            },
            allow_input_downcast=True,
            on_unused_input='warn'
        )

    def __make_test_function(self):
        X = self.vartypeX()
        return theano.function(
            inputs=[X],
            outputs=self.expression(X),
            allow_input_downcast=True
        )

    def __make_validation_function(self):
        X = self.vartypeX()
        Y = self.vartypeY()
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
            allow_input_downcast=True,
            on_unused_input='warn'
        )

def train_GAN(G,
              D,
              batch_size,
              k_steps,
              n_epochs,
              steps_per_epoch,
              data_gen,
              z_prior_gen,
              D_savepath=None,
              G_savepath=None):
    """ Train a generative adversarial network

    Arguments:
      G: the compiled generator network
      D: the compiled discriminator network
      batch_size: batch size
      k_steps: number of training steps for the discriminator
                for each generator training step
      n_epochs: number of training epochs
      steps_per_epochs: number of steps before changing epoch, where one step corresponds
                        to k_steps of discriminator training and 1 step of generator training
      data_gen: pair of python functions accepting size argument returning
                generator yielding chunks of data of requested size. (train, validation)
      z_prior_gen: same as data_gen but for prior over z-space
    """

    data_stream = data_gen[0](batch_size * k_steps)
    v_data_stream = data_gen[1](batch_size * k_steps)

    d_z_stream = z_prior_gen[0](batch_size * k_steps)
    v_d_z_stream = z_prior_gen[1](batch_size * k_steps)

    
    z_stream = z_prior_gen[0](batch_size)
    v_z_stream = z_prior_gen[1](batch_size)
    
    for epoch in range(n_epochs):
        log.warning('epoch %d', epoch)
        for i in range(steps_per_epoch):

            X = data_stream.next()
            GZ = G(d_z_stream.next())

            VX = v_data_stream.next()
            VGZ = G(v_d_z_stream.next())

            D.train(
                X=X,
                Y=GZ,
                val_data=(VX, VGZ),
                n_epochs=1,
                start_epoch=epoch,
                savepath=D_savepath
            )

            Z = z_gen.next()
            VZ = v_z_gen.next()
            G.train(
                X=Z,
                Y=G(Z),
                val_data=(Z,G(VZ)),
                n_epochs=1,
                start_epoch=epoch,
                savepath=G_savepath
            )

def get_pickled_attr(grp, key, default=None):
    if key in grp.attrs:
        return cPickle.loads(grp.attrs[key].tostring())
    else:
        return default
        
