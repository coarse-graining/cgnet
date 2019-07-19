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

    np.testing.assert_allclose(feature_dist_mean,
                               stats.stats_dict['Distances']['mean'],
                               rtol=1e-4)
    np.testing.assert_allclose(feature_dist_std,
                               stats.stats_dict['Distances']['std'],
                               rtol=1e-4)


def test_angle_statistics():
    # Make sure angle statistics are consistent with numpy

    feature_angle_mean = np.mean(f.angles.numpy(), axis=0)
    feature_angle_std = np.std(f.angles.numpy(), axis=0)

    np.testing.assert_allclose(feature_angle_mean,
                               stats.stats_dict['Angles']['mean'], rtol=1e-5)
    np.testing.assert_allclose(feature_angle_std,
                               stats.stats_dict['Angles']['std'], rtol=1e-5)


def test_dihedral_statistics():
    # Make sure dihedral statistics are consistent with numpy

    feature_dihed_cos_mean = np.mean(f.dihedral_cosines.numpy(), axis=0)
    feature_dihed_cos_std = np.std(f.dihedral_cosines.numpy(), axis=0)
    feature_dihed_sin_mean = np.mean(f.dihedral_sines.numpy(), axis=0)
    feature_dihed_sin_std = np.std(f.dihedral_sines.numpy(), axis=0)

    np.testing.assert_allclose(feature_dihed_cos_mean,
                               stats.stats_dict['Dihedral_cosines']['mean'],
                               rtol=1e-6)
    np.testing.assert_allclose(feature_dihed_cos_std,
                               stats.stats_dict['Dihedral_cosines']['std'],
                               rtol=1e-6)

    np.testing.assert_allclose(feature_dihed_sin_mean,
                               stats.stats_dict['Dihedral_sines']['mean'],
                               rtol=1e-6)
    np.testing.assert_allclose(feature_dihed_sin_std,
                               stats.stats_dict['Dihedral_sines']['std'],
                               rtol=1e-6)


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

    # Notes
    # -----
    # Sometimes this raises a RuntimeWarning about dividing by zero when
    # the bond constant attribute kb is zero. Perhaps this is due to
    # creating random features.

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


def test_form_redundant_distances():
    # Test to se if the redundant distance idex matrix is formed properly
    stats.form_redundant_distances()
    index_mapping = stats.redundant_distance_mapping
    assert index_mapping.shape == (beads, beads - 1)
    # mock distance data
    dist = np.random.randn(frames, int((beads - 1) * (beads) / 2))
    redundant_dist =  dist[:, index_mapping]
    assert redundant_dist.shape == (frames, beads, beads - 1)
