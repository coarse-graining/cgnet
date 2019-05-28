import numpy as np
import scipy.spatial
import torch

from cgtools.feature import ProteinBackboneFeature

frames = np.random.randint(10)
beads = np.random.randint(4, 10)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)


def test_distance_features():
    """Make sure pairwise distance features are consistent with scipy"""

    f = ProteinBackboneFeature()
    out = f.forward(xt)

    Dmat_x0 = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(x[0]))

    x0_feature_distances = list(f.distances[0][:].numpy())
    feature_descriptions = f.descriptions['Distances']

    x0_scipy_distances = [Dmat_x0[feature_descriptions[i]]
                          for i in range(len(feature_descriptions))]

    np.testing.assert_array_almost_equal(x0_feature_distances,
                                         x0_scipy_distances)


def test_angle_features():
    """Make sure pairwise angle features are consistent with manual calculation"""

    f = ProteinBackboneFeature()
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

    np.testing.assert_array_almost_equal(f.angles, angles, decimal=5)


def test_dihedral_features():
    """Make sure pairwise dihedral features are consistent with manual calculation"""
    
    f = ProteinBackboneFeature()
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
    np.testing.assert_array_almost_equal(np.abs(feature_diheds),
                                         np.abs(diheds), decimal=4)
