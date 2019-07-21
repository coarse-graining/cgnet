# Author: Brooke Husic
# Contributors : Nick Charron

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


def test_idx_functions():
    # Test proper retrieval of feature indices
    nums = [len(stats.descriptions[feat_name]) for feat_name in stats.order]
    dist_idx = stats.return_indices('Distances')
    start_idx = 0
    for num, desc in zip(nums, stats.order):
        if 'Distances' == desc:
            break
        else:
            start_idx += num
    indices = range(0, len(stats.descriptions['Distances']))
    indices = [idx + start_idx for idx in indices]
    assert len(dist_idx) == (beads) * (beads - 1) / 2
    assert dist_idx == indices

    ang_idx = stats.return_indices('Angles')
    start_idx = 0
    for num, desc in zip(nums, stats.order):
        if 'Angles' == desc:
            break
        else:
            start_idx += num
    indices = range(0, len(stats.descriptions['Angles']))
    indices = [idx + start_idx for idx in indices]
    assert len(ang_idx) == beads - 2
    assert ang_idx == indices

    dihedral_sin_idx = stats.return_indices('Dihedral_sines')
    start_idx = 0
    for num, desc in zip(nums, stats.order):
        if 'Dihedral_sines' == desc:
            break
        else:
            start_idx += num
    indices = range(0, len(stats.descriptions['Dihedral_sines']))
    indices = [idx + start_idx for idx in indices]
    assert len(dihedral_sin_idx) == beads - 3
    assert dihedral_sin_idx == indices

    dihedral_cos_idx = stats.return_indices('Dihedral_cosines')
    start_idx = 0
    for num, desc in zip(nums, stats.order):
        if 'Dihedral_cosines' == desc:
            break
        else:
            start_idx += num
    indices = range(0, len(stats.descriptions['Dihedral_cosines']))
    indices = [idx + start_idx for idx in indices]
    assert len(dihedral_cos_idx) == beads - 3
    assert dihedral_cos_idx == indices

    bond_idx = stats.return_indices('Bonds')
    start_idx = 0
    for num, desc in zip(nums, stats.order):
        if 'Distances' == desc:
            break
        else:
            start_idx += num
    indices = [stats.descriptions['Distances'].index(pair)
               for pair in stats._adj_pairs]
    indices = [idx + start_idx for idx in indices]

    assert len(bond_idx) == beads - 1
    assert bond_idx == indices
