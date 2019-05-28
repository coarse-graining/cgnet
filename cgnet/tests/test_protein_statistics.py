# Author: Brooke Husic

import numpy as np
import scipy.spatial
import torch

from cgtools.feature import ProteinBackboneFeature
from cgtools.feature import ProteinBackboneStatistics

frames = np.random.randint(10)
beads = np.random.randint(4, 10)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)

f = ProteinBackboneFeature()
out = f.forward(xt)
stats = ProteinBackboneStatistics(xt)


def test_distance_statistics():
    """Make sure distance statistics are consistent with numpy"""

    feature_dist_mean = np.mean(f.distances.numpy(), axis=0)
    feature_dist_std = np.std(f.distances.numpy(), axis=0)

    np.testing.assert_array_almost_equal(feature_dist_mean,
                                         stats.stats_dict['Distances']['mean'])
    np.testing.assert_array_almost_equal(feature_dist_std,
                                         stats.stats_dict['Distances']['std'])


def test_angle_statistics():
    """Make sure angle statistics are consistent with numpy"""

    feature_angle_mean = np.mean(f.angles.numpy(), axis=0)
    feature_angle_std = np.std(f.angles.numpy(), axis=0)

    np.testing.assert_array_almost_equal(feature_angle_mean,
                                         stats.stats_dict['Angles']['mean'])
    np.testing.assert_array_almost_equal(feature_angle_std,
                                         stats.stats_dict['Angles']['std'])


def test_dihedral_statistics():
    """Make sure dihedral statistics are consistent with numpy"""

    feature_dihed_cos_mean = np.mean(f.dihedral_cosines.numpy(), axis=0)
    feature_dihed_cos_std = np.std(f.dihedral_cosines.numpy(), axis=0)
    feature_dihed_sin_mean = np.mean(f.dihedral_sines.numpy(), axis=0)
    feature_dihed_sin_std = np.std(f.dihedral_sines.numpy(), axis=0)

    np.testing.assert_array_almost_equal(feature_dihed_cos_mean,
                                         stats.stats_dict['Dihedral_cosines']['mean'])
    np.testing.assert_array_almost_equal(feature_dihed_cos_std,
                                         stats.stats_dict['Dihedral_cosines']['std'])

    np.testing.assert_array_almost_equal(feature_dihed_sin_mean,
                                         stats.stats_dict['Dihedral_sines']['mean'])
    np.testing.assert_array_almost_equal(feature_dihed_sin_std,
                                         stats.stats_dict['Dihedral_sines']['std'])
