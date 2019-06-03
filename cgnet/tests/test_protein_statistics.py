# Author: Brooke Husic

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import ProteinBackboneFeature
from cgnet.feature import ProteinBackboneStatistics

frames = np.random.randint(1, 10)
beads = np.random.randint(4, 10)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)

f = ProteinBackboneFeature()
out = f.forward(xt)
stats = ProteinBackboneStatistics(xt)


def test_distance_statistics():
    # Make sure distance statistics are consistent with numpy

    feature_dist_mean = np.mean(f.distances.numpy(), axis=0)
    feature_dist_std = np.std(f.distances.numpy(), axis=0)

    np.testing.assert_array_almost_equal(feature_dist_mean,
                                         stats.stats_dict['Distances']['mean'])
    np.testing.assert_array_almost_equal(feature_dist_std,
                                         stats.stats_dict['Distances']['std'])


def test_angle_statistics():
    # Make sure angle statistics are consistent with numpy

    feature_angle_mean = np.mean(f.angles.numpy(), axis=0)
    feature_angle_std = np.std(f.angles.numpy(), axis=0)

    np.testing.assert_array_almost_equal(feature_angle_mean,
                                         stats.stats_dict['Angles']['mean'])
    np.testing.assert_array_almost_equal(feature_angle_std,
                                         stats.stats_dict['Angles']['std'])


def test_dihedral_statistics():
    # Make sure dihedral statistics are consistent with numpy

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


def test_zscore_dict_1():
    # Make sure the "flipped" zscore dict has the right structure
    zscore_dict = stats.get_zscores(flip_dict=True)
    n_keys = beads*(beads-1)/2 + beads-2 + 2*(beads-3)

    assert len(zscore_dict) == n_keys


def test_zscore_dict_2():
    # Make sure the zscore dict has the right structure
    zscore_dict = stats.get_zscores(flip_dict=False)
    n_keys = beads*(beads-1)/2 + beads-2 + 2*(beads-3)

    for k in zscore_dict.keys():
        assert len(zscore_dict[k]) == n_keys


def test_bondconst_dict_1():
    # Make sure the "flipped" bond constant dict has the right structure
    bondconst_dict = stats.get_bond_constants(flip_dict=True)
    n_keys = beads*(beads-1)/2 + beads-2 + 2*(beads-3)

    assert len(bondconst_dict) == n_keys


def test_bondconst_dict_2():
    # Make sure the bond constant dict has the right structure
    bondconst_dict = stats.get_bond_constants(flip_dict=False)
    n_keys = beads*(beads-1)/2 + beads-2 + 2*(beads-3)
    n_keys_bondconst = beads-1 + beads-2

    for k in bondconst_dict.keys():
        if k == 'k':
            assert len(bondconst_dict[k]) == n_keys_bondconst
        else:
            assert len(bondconst_dict[k]) == n_keys
