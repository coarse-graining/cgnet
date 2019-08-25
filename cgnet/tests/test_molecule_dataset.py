# Author: Brooke Husic

import numpy as np
import torch

from cgnet.feature import MoleculeDataset
from nose.exc import SkipTest

# We create an artificial dataset with a random number of 
# frames, beads, and dimensions. Since we aren't actually
# doing any featurization, we can use an arbitrary number
# of dimensions

# For some tests we want an even number of frames
frames = np.random.randint(1, 10)*2

beads = np.random.randint(1, 10)
dims = np.random.randint(1, 5)

coords = np.random.randn(frames, beads, dims) # e.g. coords
forces = np.random.randn(frames, beads, dims) # e.g. forces


def test_adding_data():
    # Make sure data is added correctly to a dataset

    # Build a dataset with all the data
    ds1 = MoleculeDataset(coords, forces)

    # Build a dataset with the first half of the data...
    ds2 = MoleculeDataset(coords, forces, selection=np.arange(frames//2))
    # ... then add the second half afterward
    ds2.add_data(coords, forces, selection=np.arange(frames//2, frames))

    # Make sure they're the same
    np.testing.assert_array_equal(ds1.coordinates, ds2.coordinates)
    np.testing.assert_array_equal(ds1.forces, ds2.forces)


def test_stride():
    # Make sure dataset stride returns correct results

    stride = np.random.randint(1, 4)
    ds = MoleculeDataset(coords, forces, stride=stride)

    strided_coords = coords[::stride]
    strided_forces = forces[::stride]

    np.testing.assert_array_equal(ds.coordinates, strided_coords)
    np.testing.assert_array_equal(ds.forces, strided_forces)


def test_indexing():
    # Make sure dataset indexing works

    # Make a random slice with possible repeats
    selection = [np.random.randint(frames)
                 for _ in range(np.random.randint(frames))]
    ds = MoleculeDataset(coords, forces)

    coords_tensor_from_numpy = torch.from_numpy(coords[selection])
    forces_tensor_from_numpy = torch.from_numpy(forces[selection])
    coords_tensor_from_ds, forces_tensor_from_ds = ds[selection]

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
        selection = np.random.randint(20)
        ds = MoleculeDataset(x, y, device=torch.device('cuda'))
        np.testing.assert_equal(ds[selection][0].device.type, 'cuda')
        np.testing.assert_equal(ds[selection][1].device.type, 'cuda')
