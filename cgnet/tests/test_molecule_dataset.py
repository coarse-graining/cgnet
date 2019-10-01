# Author: Brooke Husic

import numpy as np
import torch
import warnings

from cgnet.feature import MoleculeDataset

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
    # Make sure dataset indexing works (no embeddings)

    # Make a random slice with possible repeats
    selection = [np.random.randint(frames)
                 for _ in range(np.random.randint(frames))]
    ds = MoleculeDataset(coords, forces)

    coords_tensor_from_numpy = torch.from_numpy(coords[selection])
    forces_tensor_from_numpy = torch.from_numpy(forces[selection])
    # The third argument is an empty tensor because no embeddings have been
    # specified
    coords_tensor_from_ds, forces_tensor_from_ds, empty_tensor = ds[selection]

    assert coords_tensor_from_ds.requires_grad
    np.testing.assert_array_equal(coords_tensor_from_numpy,
                                  coords_tensor_from_ds.detach().numpy())
    np.testing.assert_array_equal(forces_tensor_from_numpy,
                                  forces_tensor_from_ds.detach().numpy())
    assert len(empty_tensor) == 0


def test_one_dimensional_embedding_shape():
    # Test reshaping of integer embeddings of shape (n_frames, n_beads)
    embeddings = np.abs(np.floor(np.random.randn(frames, beads))).astype(int)

    # This will raise a warning to reshape the embeddings unless we suppress
    # them, but it should work
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = MoleculeDataset(coords, forces, embeddings)

    # Test that the embeddings shape output from the ds object is appropriate
    assert ds[:][2].shape == (frames, beads, 1)


def test_multi_dimensional_embedding_shape():
    # Test shape of multidimensional embeddings
    n_properties = np.random.randint(2, 5)
    embeddings = np.abs(np.floor(np.random.randn(
        frames, beads, n_properties))).astype(int)

    ds = MoleculeDataset(coords, forces, embeddings)

    assert ds[:][2].shape == (frames, beads, n_properties)
