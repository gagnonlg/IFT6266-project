""" Densely connected multi-layer perceptron """

# pylint: disable=invalid-name, no-member, too-few-public-methods

import numpy as np
import theano
import theano.tensor as T


class LinearTransformation(object):
    """ Linear transformation of the form X * W + b """

    def __init__(self, shape):
        """ Linear transformation of the form X * W + b

        Arguments:
          shape : tuple (n_in, n_out) defining the dimensions of W
        """

        self.shape = shape
        n_in, n_out = self.shape

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(1.0 / n_in),
                high=-np.sqrt(1.0 / n_in),
                size=(n_in, n_out),
            ).astype('float32'),
            name='W',
        )

        self.b = theano.shared(
            value=np.zeros(n_out).astype('float32'),
            name='b',
        )

        # self.output = T.dot(X, self.W) + self.b

        self.parameters = [self.W, self.b]

    def __call__(self, X):
        """ Create a symbolic expression X * W + b """
        return T.dot(X, self.W) + self.b
