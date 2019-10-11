# Author: Brooke Husic
# Contributors: Dominik Lemm

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryFeature, Geometry
g = Geometry(method='torch')

# The following sets up our pseudo-simulation data

# Number of frames
frames = np.random.randint(1, 10)

# Number of coarse-grained beads. We need at least 4 so we can do dihedrals.
beads = np.random.randint(8, 10)

# Number of dimensions; for now geometry only handles 3
dims = 3

# Create a pseudo simulation dataset
data = np.random.randn(frames, beads, dims)
data_tensor = torch.Tensor(data)

# Note: currently get_distance_indices is not directly tested.
# Possibly add a test here?
distance_inds, _ = g.get_distance_indices(beads)

angle_inds = [(i, i+1, i+2) for i in range(beads-2)]
dihedral_inds = [(i, i+1, i+2, i+3) for i in range(beads-3)]


def test_distance_features():
    # Make sure pairwise distance features are consistent with scipy

    geom_feature = GeometryFeature(n_beads=beads)
    # Forward pass calculates features (distances, angles, dihedrals)
    # and makes them accessible as attributes
    _ = geom_feature.forward(data_tensor)

    # Test each frame x_i
    for frame_ind in range(frames):
        Dmat_xi = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(data[frame_ind]))

        xi_feature_distances = list(geom_feature.distances[frame_ind].numpy())
        feature_descriptions = geom_feature.descriptions['Distances']

        # Arrange the scipy distances in the right order for comparing
        # to the GeometryFeature distances
        xi_scipy_distances = [Dmat_xi[feature_descriptions[i]]
                              for i in range(len(feature_descriptions))]

        np.testing.assert_allclose(xi_feature_distances,
                                   xi_scipy_distances, rtol=1e-6)


def test_backbone_angle_features():
    # Make sure backbone angle features are consistent with manual calculation

    # For spatial coordinates a, b, c, the angle \theta describing a-b-c
    # is calculated using the following formula:
    #
    # \overline{ba} = a - b
    # \overline{cb} = c - b
    # \cos(\theta) = (\frac{\overline{ba} \dot \overline{cb}}
    #                      {||\overline{ba}|| ||\overline{cb}||}
    # \theta = \arccos(\theta)

    geom_feature = GeometryFeature(n_beads=beads)
    # Forward pass calculates features (distances, angles, dihedrals)
    # and makes them accessible as attributes
    _ = geom_feature.forward(data_tensor)

    # Manually calculate the angles one frame at a time
    angles = []
    for frame_data in data:
        angle_list = []
        for i in range(data.shape[1] - 2):
            a = frame_data[i]
            b = frame_data[i+1]
            c = frame_data[i+2]

            ba = a - b
            cb = c - b

            cos_angle = np.dot(ba, cb) / (np.linalg.norm(ba)
                                          * np.linalg.norm(cb))
            angle = np.arccos(cos_angle)
            angle_list.append(angle)
        angles.append(angle_list)

    np.testing.assert_allclose(geom_feature.angles, angles, rtol=1e-4)


def test_dihedral_features():
    # Make sure backbone dihedral features are consistent with manual calculation

    # For spatial coordinates a, b, c, d, the dihedral \alpha describing
    # a-b-c-d (i.e., the plane between angles a-b-c- and b-c-d-) is calculated
    # using the following formula:
    #
    # \overline{ba} = b - a
    # \overline{cb} = c - a
    # \overline{dc} = d - c
    #
    # % normal vector with plane of first and second angles, respectively
    # n_1 = \overline{ba} \times \overline{cb} 
    # n_2 = \overline{cb} \ times \overline{dc}
    #
    # m_1 = n_2 \times n_1
    # 
    # \sin(\alpha) = \frac{m_1 \dot \overline{cb}}
    #                     {\sqrt{\overline{cb} \dot \overline{cb}}}
    # \cos(\alpha) = n_2 \dot n_1
    # \alpha = \arctan{\frac{\sin(\alpha)}{\cos(\alpha)}}

    geom_feature = GeometryFeature(n_beads=beads)
    # Forward pass calculates features (distances, angles, dihedrals)
    # and makes them accessible as attributes
    _ = geom_feature.forward(data_tensor)

    # Manually calculate the dihedrals one frame at a time
    diheds = []
    for frame_data in data:
        dihed_list = []
        for i in range(data.shape[1] - 3):
            a = frame_data[i]
            b = frame_data[i+1]
            c = frame_data[i+2]
            d = frame_data[i+3]

            ba = b-a
            cb = c-b
            dc = d-c

            n1 = np.cross(ba, cb)
            n2 = np.cross(cb, dc)
            m1 = np.cross(n2, n1)
            term1 = np.dot(m1, cb)/np.sqrt(np.dot(cb, cb))
            term2 = np.dot(n2, n1)
            dihed_list.append(np.arctan2(term1, term2))
        diheds.append(dihed_list)

    # Instead of comparing the sines and cosines, compare the arctans
    feature_diheds = [np.arctan2(geom_feature.dihedral_sines[i].numpy(),
                                 geom_feature.dihedral_cosines[i].numpy())
                      for i in range(len(geom_feature.dihedral_sines))]
    np.testing.assert_allclose(np.abs(feature_diheds),
                               np.abs(diheds), rtol=1e-4)


def test_distance_index_shuffling():
    # Make sure shuffled distances return the right results

    # Create a dataset with one frame, 10 beads, 3 dimensions
    data_to_shuffle = np.random.randn(1, 10, 3)
    data_to_shuffle_tensor = torch.Tensor(data_to_shuffle)

    y_dist_inds, _ = g.get_distance_indices(10)

    geom_feature = GeometryFeature(feature_tuples=y_dist_inds)
    # Forward pass calculates features (distances, angles, dihedrals)
    # and makes them accessible as attributes
    _ = geom_feature.forward(data_to_shuffle_tensor)

    # Shuffle the distances indices
    inds = np.arange(len(y_dist_inds))
    np.random.shuffle(inds)

    shuffled_inds = [tuple(i) for i in np.array(y_dist_inds)[inds]]
    geom_feature_shuffle = GeometryFeature(feature_tuples=shuffled_inds)
    _ = geom_feature_shuffle.forward(data_to_shuffle_tensor)

    # See if the non-shuffled distances are the same when indexexed according
    # to the shuffling
    np.testing.assert_array_equal(geom_feature_shuffle.distances[0],
                                  geom_feature.distances[0][inds])


def test_angle_index_shuffling():
    # Make sure shuffled angles return the right results

    # Create a dataset with one frame, 100 beads, 3 dimensions
    data_to_shuffle = np.random.randn(1, 100, 3)
    data_to_shuffle_tensor = torch.Tensor(data_to_shuffle)

    y_angle_inds = [(i, i+1, i+2) for i in range(100-2)]

    geom_feature = GeometryFeature(feature_tuples=y_angle_inds)
    # Forward pass calculates features (distances, angles, dihedrals)
    # and makes them accessible as attributes
    _ = geom_feature.forward(data_to_shuffle_tensor)

    # Shuffle all the inds that can serve as an angle start
    inds = np.arange(100-2)
    np.random.shuffle(inds)

    shuffled_inds = [tuple(i) for i in np.array(y_angle_inds)[inds]]
    geom_feature_shuffle = GeometryFeature(feature_tuples=shuffled_inds)
    _ = geom_feature_shuffle.forward(data_to_shuffle_tensor)

    # See if the non-shuffled angles are the same when indexexed according
    # to the shuffling
    np.testing.assert_array_equal(geom_feature_shuffle.angles[0],
                                  geom_feature.angles[0][inds])


def test_dihedral_index_shuffling():
    # Make sure shuffled dihedrals return the right results

    # Create a dataset with one frame, 100 beads, 3 dimensions
    data_to_shuffle = np.random.randn(1, 100, 3)
    data_to_shuffle_tensor = torch.Tensor(data_to_shuffle)

    y_dihed_inds = [(i, i+1, i+2, i+3) for i in range(100-3)]

    geom_feature = GeometryFeature(feature_tuples=y_dihed_inds)
    # Forward pass calculates features (distances, angles, dihedrals)
    # and makes them accessible as attributes
    _ = geom_feature.forward(data_to_shuffle_tensor)

    # Shuffle all the inds that can serve as a dihedral start
    inds = np.arange(100-3)
    np.random.shuffle(inds)

    shuffled_inds = [tuple(i) for i in np.array(y_dihed_inds)[inds]]
    geom_feature_shuffle = GeometryFeature(feature_tuples=shuffled_inds)
    _ = geom_feature_shuffle.forward(data_to_shuffle_tensor)

    # See if the non-shuffled dihedral sines and cosines are the same when
    # indexexed according to the shuffling
    np.testing.assert_allclose(geom_feature_shuffle.dihedral_cosines[0],
                               geom_feature.dihedral_cosines[0][inds], rtol=1e-5)

    np.testing.assert_allclose(geom_feature_shuffle.dihedral_sines[0],
                               geom_feature.dihedral_sines[0][inds], rtol=1e-5)
