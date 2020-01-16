# Author: Brooke Husic
# Contributors: Dominik Lemm

import numpy as np
import torch

from cgnet.feature import Geometry

g_numpy = Geometry(method='numpy')
g_torch = Geometry(method='torch')

# Define sizes for a pseudo-dataset
frames = np.random.randint(10, 30)
beads = np.random.randint(5, 10)

# create random linear protein data
coords = np.random.randn(frames, beads, 3).astype(np.float32)

# Calculate redundant distances and create a simple neighbor list in which all
# beads see each other (shape [n_frames, n_beads, n_beads -1]).
_distance_pairs, _ = g_numpy.get_distance_indices(beads, [], [])
redundant_distance_mapping = g_numpy.get_redundant_distance_mapping(
    _distance_pairs)

neighbor_cutoff = np.random.uniform(0, 3)


def test_tile_methods_numpy_vs_torch():
    # Test to make sure geometry.tile is still equivalent between numpy
    # and pytorch

    # Create inputs for a 3d array that will have 24 elements
    A = np.array([np.random.randint(10) for _ in range(24)])

    # Make two likely different shapes for the array and the tiling
    # with friendly factors
    shape_one = [2, 3, 4]
    np.random.shuffle(shape_one)

    shape_two = [2, 3, 4]
    np.random.shuffle(shape_two)

    # Reshape A with the first shape
    A = A.reshape(*shape_one).astype(np.float32)

    # Test whether the tiling is equivalent to the second shape
    # Add in the standard check for fun
    g_numpy.check_array_vs_tensor(A)
    tile_numpy = g_numpy.tile(A, shape_two)

    g_torch.check_array_vs_tensor(torch.Tensor(A))
    tile_torch = g_torch.tile(torch.Tensor(A), shape_two)

    np.testing.assert_array_equal(tile_numpy, tile_torch)


def test_distances_and_neighbors_numpy_vs_torch():
    # Comparison of numpy and torch outputs for getting geometry.get_distances
    # and geometry.get_neighbors

    # Calculate distances, neighbors, and neighbor mask using the numpy
    # version of Geometry
    distances_numpy = g_numpy.get_distances(_distance_pairs,
                                            coords,
                                            norm=True)
    distances_numpy = distances_numpy[:, redundant_distance_mapping]
    neighbors_numpy, neighbors_mask_numpy = g_numpy.get_neighbors(
        distances_numpy,
        cutoff=neighbor_cutoff)

    # Calculate distances, neighbors, and neighbor mask using the torch
    # version of Geometry
    distances_torch = g_torch.get_distances(_distance_pairs,
                                            torch.from_numpy(coords),
                                            norm=True)
    distances_torch = distances_torch[:, redundant_distance_mapping]
    neighbors_torch, neighbors_mask_torch = g_torch.get_neighbors(
        distances_torch,
        cutoff=neighbor_cutoff)

    np.testing.assert_array_equal(distances_numpy, distances_torch)
    np.testing.assert_array_equal(neighbors_numpy, neighbors_torch)
    np.testing.assert_array_equal(neighbors_mask_numpy, neighbors_mask_torch)


def test_hide_dummy_atoms_numpy():
    # Test that any atoms with embeddings of zero don't show up in
    # the neighbor list (numpy version)

    # Calculate distances, neighbors, and neighbor mask using the numpy
    # version of Geometry
    distances_numpy = g_numpy.get_distances(_distance_pairs,
                                            coords,
                                            norm=True)
    distances_numpy = distances_numpy[:, redundant_distance_mapping]
    neighbors_numpy, neighbors_mask_numpy = g_numpy.get_neighbors(
        distances_numpy,
        cutoff=neighbor_cutoff)

    possible_neighbors = np.unique(neighbors_numpy)
    if 0 in possible_neighbors:
        possible_neighbors = possible_neighbors[1:]

    np.random.shuffle(possible_neighbors)
    dummy_atoms = possible_neighbors[:3]

    n_embeddings = np.random.randint(3, 5)
    embedding_property = np.random.randint(low=1, high=n_embeddings,
                                       size=(frames, beads))

    embedding_property[:, dummy_atoms] = 0

    new_neighbors_mask_numpy = g_numpy.hide_dummy_atoms(
                                             embedding_property,
                                             neighbors_numpy,
                                             neighbors_mask_numpy)

    masked_neighbors_numpy = np.copy(neighbors_numpy)
    masked_neighbors_numpy[~g_numpy.to_type(
                                    new_neighbors_mask_numpy, g_numpy.bool)] = -1

    assert len(np.intersect1d(dummy_atoms, np.unique(masked_neighbors_numpy))) == 0


def test_hide_dummy_atoms_torch():
    # Test that any atoms with embeddings of zero don't show up in
    # the neighbor list (torch version)

    # Calculate distances, neighbors, and neighbor mask using the numpy
    # version of Geometry
    distances_torch = g_torch.get_distances(_distance_pairs,
                                            torch.from_numpy(coords),
                                            norm=True)
    distances_torch = distances_torch[:, redundant_distance_mapping]
    neighbors_torch, neighbors_mask_torch = g_torch.get_neighbors(
        distances_torch,
        cutoff=neighbor_cutoff)

    possible_neighbors = np.unique(neighbors_torch)
    if 0 in possible_neighbors:
        possible_neighbors = possible_neighbors[1:]

    np.random.shuffle(possible_neighbors)
    dummy_atoms = possible_neighbors[:3]

    n_embeddings = np.random.randint(3, 5)
    embedding_property = torch.randint(low=1, high=n_embeddings,
                                       size=(frames, beads))

    embedding_property[:, dummy_atoms] = 0

    new_neighbors_mask_torch = g_torch.hide_dummy_atoms(
                                             embedding_property,
                                             neighbors_torch,
                                             neighbors_mask_torch)

    masked_neighbors_torch = neighbors_torch.clone()
    masked_neighbors_torch[~g_torch.to_type(
                                    new_neighbors_mask_torch, g_torch.bool)] = -1

    assert len(np.intersect1d(dummy_atoms, np.unique(masked_neighbors_torch))) == 0

def test_nan_check():
    # Test if an assert is raised during the computation of distances, angles
    # and dihedrals

    # Calculate angle and dihedral pair indices
    angle_pairs = [(i, i+1, i+2) for i in range(beads - 2)]
    dihedral_pairs = [(i, i+1, i+2, i+3) for i in range(beads - 3)]

    # Select random frame and bead to set NaN
    random_frame = np.random.randint(0, frames)
    random_bead = np.random.randint(0, beads)

    # Create test coordinates that contain NaN
    nan_coords = coords.copy()
    nan_coords[random_frame][random_bead] = np.nan
    torch_nan_coords = torch.from_numpy(nan_coords)

    # Check if an assert is raised
    np.testing.assert_raises(ValueError,
                             g_numpy.get_distances, _distance_pairs, nan_coords)
    np.testing.assert_raises(ValueError,
                             g_numpy.get_angles, angle_pairs, nan_coords)
    np.testing.assert_raises(ValueError,
                             g_numpy.get_dihedrals, dihedral_pairs, nan_coords)

    np.testing.assert_raises(ValueError,
                             g_torch.get_distances, _distance_pairs,
                             torch_nan_coords)
    np.testing.assert_raises(ValueError,
                             g_torch.get_angles, angle_pairs,
                             torch_nan_coords)
    np.testing.assert_raises(ValueError,
                             g_torch.get_dihedrals, dihedral_pairs,
                             torch_nan_coords)
