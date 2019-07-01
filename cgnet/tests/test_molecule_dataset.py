# Author: Brooke Husic

import numpy as np
import torch

from cgnet.feature import MoleculeDataset
from nose.exc import SkipTest

beads = np.random.randint(1, 10)
dims = np.random.randint(1, 5)

x = np.random.randn(20, beads, dims)
y = np.random.randn(20, beads, dims)


def test_adding_data():
    # Make sure data is added correctly to a dataset

    ds1 = MoleculeDataset(x, y)

    ds2 = MoleculeDataset(x, y, selection=np.arange(10))
    ds2.add_data(x, y, selection=np.arange(10, 20))

    np.testing.assert_array_equal(ds1.coordinates, ds2.coordinates)
    np.testing.assert_array_equal(ds1.forces, ds2.forces)


def test_stride():
    # Make sure dataset stride returns correct results

    stride = np.random.randint(1, 4)
    ds = MoleculeDataset(x, y, stride=stride)

    strided_x = x[::stride]
    strided_y = y[::stride]

    np.testing.assert_array_equal(ds.coordinates, strided_x)
    np.testing.assert_array_equal(ds.forces, strided_y)


def test_indexing():
    # Make sure dataset indexing works

    selection = np.random.randint(20)
    ds = MoleculeDataset(x, y)

    xt_from_numpy = torch.from_numpy(x[selection])
    xt_from_ds = ds[selection][0]

    assert xt_from_ds.requires_grad
    np.testing.assert_array_equal(xt_from_numpy, xt_from_ds.detach().numpy())


def test_cpu_mount():
    # Make sure tensors are being mapped to cpu

    selection = np.random.randint(20)
    ds = MoleculeDataset(x, y)

    np.testing.assert_equal(ds[selection][0].device.type, 'cpu')
    np.testing.assert_equal(ds[selection][1].device.type, 'cpu')


def test_gpu_mount():
    # Make sure tensors are being mapped to gpu
    if not torch.cuda.is_available():
        raise SkipTest('GPU not available for testing.')
    else:
        ds = MoleculeDataset(x, y, cuda=torch.device('cuda'))
        np.testing.assert_equal(ds[selection][0].device.type, 'cuda')
        np.testing.assert_equal(ds[selection][1].device.type, 'cuda')
