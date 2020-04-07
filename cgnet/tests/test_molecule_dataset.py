# Author: Brooke Husic

import numpy as np
import torch

from cgnet.feature import MoleculeDataset, MultiMoleculeDataset

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

# This data is used to test MultiMoleculeDataset methods
# it consists of random data for n_frames number of molecules
# with variying bead numbers per frame/example
max_beads = 20 # largest molecule size of the varaible dataset entries
variable_beads = np.random.randint(3, max_beads, size=frames) # random protein sizes
variable_coords = [np.random.randn(bead, 3) for bead in variable_beads] # random coords for each size
variable_forces = [np.random.randn(bead, 3) for bead in variable_beads] # random forces for each size
variable_embeddings = [np.random.randint(1,
                       high=max_beads, size=bead) for bead in variable_beads] # random embeddings for each size

def test_adding_data():
    # Make sure data is added correctly to a MoleculeDataset

    # Build a dataset with all the data
    ds1 = MoleculeDataset(coords, forces)

    # Build a dataset with the first half of the data...
    ds2 = MoleculeDataset(coords, forces, selection=np.arange(frames//2))
    # ... then add the second half afterward
    ds2.add_data(coords, forces, selection=np.arange(frames//2, frames))

    # Make sure they're the same
    np.testing.assert_array_equal(ds1.coordinates, ds2.coordinates)
    np.testing.assert_array_equal(ds1.forces, ds2.forces)

def test_adding_variable_data():
    # Make sure data is added correctly to a MultiMoleculeDataset

    # Build a dataset with all the data
    ds1 = MultiMoleculeDataset(variable_coords, variable_forces,
                               variable_embeddings)

    # Build a dataset with the first half of the data...
    print(np.arange(frames//2))
    ds2 = MultiMoleculeDataset(variable_coords, variable_forces,
                               variable_embeddings, selection=np.arange(frames//2))
    # ... then add the second half afterward
    ds2.add_data(variable_coords, variable_forces,
                 embeddings_list=variable_embeddings,
                 selection=np.arange(frames//2, frames))

    # Make sure they're the same
    np.testing.assert_array_equal(ds1.data, ds2.data)


def test_stride():
    # Make sure MoleculeDataset stride returns correct results

    stride = np.random.randint(1, 4)
    ds = MoleculeDataset(coords, forces, stride=stride)

    strided_coords = coords[::stride]
    strided_forces = forces[::stride]

    np.testing.assert_array_equal(ds.coordinates, strided_coords)
    np.testing.assert_array_equal(ds.forces, strided_forces)


def test_variable_stride():
    # Make sure MultiMoleculeDataset stride returns correct results

    stride = np.random.randint(1, 4)
    ds = MultiMoleculeDataset(variable_coords, variable_forces,
                              variable_embeddings, stride=stride)

    strided_coords = variable_coords[::stride]
    strided_forces = variable_forces[::stride]
    strided_embeddings = variable_embeddings[::stride]
    strided_data = [{'coords': strided_coords[i],
                     'forces': strided_forces[i],
                     'embeddings': strided_embeddings[i]}
                    for i in range(len(strided_coords))]

    np.testing.assert_array_equal(ds.data, strided_data)


def test_indexing():
    # Make sure MoleculeDataset indexing works (no embeddings)

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


def test_variable_indexing():
    # Make sure MultiMoleculeDataset indexing works (no embeddings)

    # Make a random slice with possible repeats
    selection = [np.random.randint(frames)
                 for _ in range(np.random.randint(frames))]
    ds = MultiMoleculeDataset(variable_coords, variable_forces,
                              variable_embeddings)
    manual_data = [{'coords': variable_coords[i],
                     'forces': variable_forces[i],
                     'embeddings': variable_embeddings[i]}
                    for i in selection]

    data = ds[selection]
    np.testing.assert_array_equal(manual_data, data),

def test_embedding_shape():
    # Test shape of multidimensional embeddings
    embeddings = np.random.randint(1, 10, size=(frames, beads))

    ds = MoleculeDataset(coords, forces, embeddings)

    assert ds[:][2].shape == (frames, beads)
    np.testing.assert_array_equal(ds.embeddings, embeddings)
