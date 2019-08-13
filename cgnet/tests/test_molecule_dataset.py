# Author: Brooke Husic

import numpy as np
import torch

from cgnet.feature import MoleculeDataset

# We create an artificial dataset with a random number of 
# frames, beads, and dimensions. Since we aren't actually
# doing any featurization, we can use an arbitrary number
# of dimensions

# For some tests we want an even number of frames
frames = np.random.randint(1, 10)*2

beads = np.random.randint(1, 10)
dims = np.random.randint(1, 5)

x = np.random.randn(frames, beads, dims) # e.g. coords
y = np.random.randn(frames, beads, dims) # e.g. forces


def test_adding_data():
    # Make sure data is added correctly to a dataset

    # Build a dataset with all the data
    ds1 = MoleculeDataset(x, y)

    # Build a dataset with the first half of the data...
    ds2 = MoleculeDataset(x, y, selection=np.arange(frames//2))
    # ... then add the second half afterward
    ds2.add_data(x, y, selection=np.arange(frames//2, frames))

    # Make sure they're the same
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

    # Make a random slice with possible repeats
    selection = [np.random.randint(frames) for _ in range(np.random.randint(frames))]
    ds = MoleculeDataset(x, y)

    xt_from_numpy = torch.from_numpy(x[selection])
    yt_from_numpy = torch.from_numpy(y[selection])
    xt_from_ds, yt_from_ds = ds[selection]

    assert xt_from_ds.requires_grad
    np.testing.assert_array_equal(xt_from_numpy, xt_from_ds.detach().numpy())
    np.testing.assert_array_equal(yt_from_numpy, yt_from_ds.detach().numpy())
