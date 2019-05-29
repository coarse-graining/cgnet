# Author: Brooke Husic

import numpy as np
import torch

from cgnet.feature import MoleculeDataset

beads = np.random.randint(1, 10)
dims = np.random.randint(1, 5)

x = np.random.randn(20, beads, dims)
y = np.random.randn(20, beads, dims)


def test_adding_data():
    """Make sure data is added correctly to a dataset"""
    
    ds1 = MoleculeDataset(x, y)

    ds2 = MoleculeDataset(x, y, selection=np.arange(10))
    ds2.add_data(x, y, selection=np.arange(10, 20))

    np.testing.assert_array_equal(ds1.coordinates, ds2.coordinates)
    np.testing.assert_array_equal(ds1.forces, ds2.forces)


def test_stride():
    """Make sure dataset stride returns correct results"""

    stride = np.random.randint(1, 4)
    ds = MoleculeDataset(x, y, stride=stride)

    strided_x = x[::stride]
    strided_y = y[::stride]

    np.testing.assert_array_equal(ds.coordinates, strided_x)
    np.testing.assert_array_equal(ds.forces, strided_y)
