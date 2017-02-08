import dataset
import unittest

import numpy as np

class TestImage(unittest.TestCase):

    def setUp(self):
        datadir = dataset.retrieve()
        self.path = datadir + '/train2014/COCO_train2014_000000000025.jpg'

    def test_reco(self):
        tensor = dataset.load_image(self.path)
        reco = dataset.reconstruct_from_flat(
            *dataset.get_flattened_example(tensor)
        )
        self.assertTrue(np.allclose(tensor, reco))

if __name__ == '__main__':
    unittest.main()
