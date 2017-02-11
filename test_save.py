import os
import unittest

import numpy as np

import mlp

class TestSave(unittest.TestCase):

    def setUp(self):

        self.path = '____TEST____MODEL____.h5'
        np.random.seed(124214)

    def runTest(self):
        network1 = mlp.MLP([10, 5, 3])
        network1.save(self.path)
        network2 = mlp.load(self.path)
        for i in range(len(network1.layers)):
            W1 = network1.layers[i].parameters[0].get_value()
            b1 = network1.layers[i].parameters[1].get_value()
            W2 = network2.layers[i].parameters[0].get_value()
            b2 = network2.layers[i].parameters[1].get_value()
            self.assertTrue(np.allclose(W1, W2))
            self.assertTrue(np.allclose(b1, b2))

        X = np.random.uniform(size=(10,10)).astype('float32')
        self.assertTrue(np.allclose(network1(X), network2(X)))


    def tearDown(self):
        os.remove('____TEST____MODEL____.h5')
        
