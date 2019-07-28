# Author: Brooke Husic

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryFeature, Geometry
g = Geometry(method='torch')

frames = np.random.randint(1, 10)
beads = np.random.randint(8, 10)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)

distance_inds, _ = g.get_distance_indices(beads)
angle_inds = [(i, i+1, i+2) for i in range(beads-2)]
dihedral_inds = [(i, i+1, i+2, i+3) for i in range(beads-3)]


def test_distance_features():
    # Make sure pairwise distance features are consistent with scipy

    f = GeometryFeature(n_beads=beads)
    out = f.forward(xt)

    Dmat_x0 = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(x[0]))

    x0_feature_distances = list(f.distances[0][:].numpy())
    feature_descriptions = f.descriptions['Distances']

    x0_scipy_distances = [Dmat_x0[feature_descriptions[i]]
                          for i in range(len(feature_descriptions))]

    np.testing.assert_allclose(x0_feature_distances,
                               x0_scipy_distances, rtol=1e-6)


def test_backbone_angle_features():
    # Make sure angle features are consistent with manual calculation

    f = GeometryFeature(n_beads=beads)
    out = f.forward(xt)

    angles = []
    for j, z in enumerate(x):
        angle_list = []
        for i in range(x.shape[1]-2):
            a = z[i]
            b = z[i+1]
            c = z[i+2]

            ba = b-a
            cb = c-b

            cos_angle = np.dot(ba, cb) / (np.linalg.norm(ba)
                                          * np.linalg.norm(cb))
            angle = np.arccos(cos_angle)
            angle_list.append(angle)
        angles.append(angle_list)

    np.testing.assert_allclose(f.angles, angles, rtol=1e-5)


def test_random_angle_features():
    # Make sure angle features are consistent with manual calculation

    f = GeometryFeature(n_beads=beads)
    out = f.forward(xt)

    angles = []
    for j, z in enumerate(x):
        angle_list = []
        for i in range(x.shape[1]-2):
            a = z[i]
            b = z[i+1]
            c = z[i+2]

            ba = b-a
            cb = c-b

            cos_angle = np.dot(ba, cb) / (np.linalg.norm(ba)
                                          * np.linalg.norm(cb))
            angle = np.arccos(cos_angle)
            angle_list.append(angle)
        angles.append(angle_list)

    np.testing.assert_allclose(f.angles, angles, rtol=1e-5)


def test_dihedral_features():
    # Make sure dihedral features are consistent with manual calculation

    f = GeometryFeature(n_beads=beads)
    out = f.forward(xt)

    diheds = []
    for j, z in enumerate(x):
        dihed_list = []
        for i in range(x.shape[1]-3):
            a = z[i]
            b = z[i+1]
            c = z[i+2]
            d = z[i+3]

            ba = b-a
            cb = c-b
            dc = d-c

            c1 = np.cross(ba, cb)
            c2 = np.cross(cb, dc)
            temp = np.cross(c2, c1)
            term1 = np.dot(temp, cb)/np.sqrt(np.dot(cb, cb))
            term2 = np.dot(c2, c1)
            dihed_list.append(np.arctan2(term1, term2))
        diheds.append(dihed_list)

    feature_diheds = [np.arctan2(f.dihedral_sines[i].numpy(),
                                 f.dihedral_cosines[i].numpy())
                      for i in range(len(f.dihedral_sines))]
    np.testing.assert_allclose(np.abs(feature_diheds),
                               np.abs(diheds), rtol=1e-4)


def test_distance_index_shuffling():
    # Make sure shuffled distances return the right results

    y = np.random.randn(1, 10, 3)
    yt = torch.Tensor(y)

    y_dist_inds, _ = g.get_distance_indices(10)

    f = GeometryFeature(feature_tuples=y_dist_inds)
    out = f.forward(yt)

    inds = np.arange(len(y_dist_inds))
    np.random.shuffle(inds)

    shuffled_inds = [tuple(i) for i in np.array(y_dist_inds)[inds]]
    f_shuffle = GeometryFeature(feature_tuples=shuffled_inds)
    out_shuffle = f_shuffle.forward(yt)

    np.testing.assert_array_equal(f_shuffle.distances[0], f.distances[0][inds])


def test_angle_index_shuffling():
    # Make sure shuffled angles return the right results

    y = np.random.randn(1, 100, 3)
    yt = torch.Tensor(y)

    y_angle_inds = [(i, i+1, i+2) for i in range(100-2)]

    f = GeometryFeature(feature_tuples=y_angle_inds)
    out = f.forward(yt)

    inds = np.arange(100-2)
    np.random.shuffle(inds)

    shuffled_inds = [tuple(i) for i in np.array(y_angle_inds)[inds]]
    f_shuffle = GeometryFeature(feature_tuples=shuffled_inds)
    out_shuffle = f_shuffle.forward(yt)

    np.testing.assert_array_equal(f_shuffle.angles[0], f.angles[0][inds])


def test_dihedral_index_shuffling():
    # Make sure shuffled dihedrals return the right results

    y = np.random.randn(1, 100, 3)
    yt = torch.Tensor(y)

    y_dihed_inds = [(i, i+1, i+2, i+3) for i in range(100-3)]

    f = GeometryFeature(feature_tuples=y_dihed_inds)
    out = f.forward(yt)

    inds = np.arange(100-3)
    np.random.shuffle(inds)

    shuffled_inds = [tuple(i) for i in np.array(y_dihed_inds)[inds]]
    f_shuffle = GeometryFeature(feature_tuples=shuffled_inds)
    out_shuffle = f_shuffle.forward(yt)

    np.testing.assert_allclose(f_shuffle.dihedral_cosines[0],
                               f.dihedral_cosines[0][inds], rtol=1e-5)

    np.testing.assert_allclose(f_shuffle.dihedral_sines[0],
                               f.dihedral_sines[0][inds], rtol=1e-5)
