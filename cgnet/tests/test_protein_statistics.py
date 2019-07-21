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


def test_idx_functions_1():
    # Test proper retrieval of feature indices for sizes
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats = ProteinBackboneStatistics(xt,
                                      get_distances=bool_list[0],
                                      get_angles=bool_list[1],
                                      get_dihedrals=bool_list[2])

    if bool_list[0]:
        assert len(stats.return_indices('Distances')) == (
            beads) * (beads - 1) / 2
        assert len(stats.return_indices('Bonds')) == beads - 1
    if bool_list[1]:
        assert len(stats.return_indices('Angles')) == beads - 2
    if bool_list[2]:
        assert len(stats.return_indices('Dihedral_cosines')) == beads - 3
        assert len(stats.return_indices('Dihedral_sines')) == beads - 3

    sum_feats = np.sum([len(stats.descriptions[feat_name])
                        for feat_name in stats.order])
    check_sum_feats = (bool_list[0] * (beads) * (beads - 1) / 2 +
                       bool_list[1] * (beads - 2) +
                       bool_list[2] * (beads - 3) * 2
                       )
    assert sum_feats == check_sum_feats


def test_idx_functions_2():
    # Test proper retrieval of feature indices for specific indices
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats = ProteinBackboneStatistics(xt,
                                      get_distances=bool_list[0],
                                      get_angles=bool_list[1],
                                      get_dihedrals=bool_list[2])

    num_dists = bool_list[0] * (beads) * (beads - 1) / 2
    num_angles = beads - 2
    num_diheds = beads - 3

    if bool_list[0]:
        np.testing.assert_array_equal(np.arange(0, num_dists),
                                      stats.return_indices('Distances'))

        bond_ind_list = [ind for ind, pair in enumerate(
            stats.descriptions['Distances'])
            if pair[1] - pair[0] == 1]
        np.testing.assert_array_equal(bond_ind_list,
                                      stats.return_indices('Bonds'))

    if bool_list[1]:
        angle_start = bool_list[0]*num_dists
        np.testing.assert_array_equal(np.arange(angle_start,
                                                num_angles + angle_start),
                                      stats.return_indices('Angles'))

    if bool_list[2]:
        dihedral_cos_start = bool_list[0]*num_dists + bool_list[1]*num_angles
        np.testing.assert_array_equal(np.arange(dihedral_cos_start,
                                                num_diheds + dihedral_cos_start),
                                      stats.return_indices('Dihedral_cosines'))

        dihedral_sin_start = dihedral_cos_start + num_diheds
        np.testing.assert_array_equal(np.arange(dihedral_sin_start,
                                                num_diheds + dihedral_sin_start),
                                      stats.return_indices('Dihedral_sines'))


def test_redundant_distance_mapping_shape():
    # Test to see if the redundant distance index matrix is formed properly
    index_mapping = stats.redundant_distance_mapping
    assert index_mapping.shape == (beads, beads - 1)
    # mock distance data
    dist = np.random.randn(frames, int((beads - 1) * (beads) / 2))
    redundant_dist = dist[:, index_mapping]
    assert redundant_dist.shape == (frames, beads, beads - 1)


def test_redundant_distance_mapping_vals():
    # Test to see if the redundant distance index matrix has correct values
    mapping = np.zeros((stats.n_beads, stats.n_beads - 1), dtype='uint8')
    for bead in range(stats.n_beads):
        def seq1(bead, n_beads):
            n = bead
            j = n_beads - 1
            while(True):
                yield n + j
                n = n + j
                j -= 1
        max_calls = stats.n_beads - bead - 1
        gen = seq1(bead, stats.n_beads)
        idx = np.array([bead] + [next(gen) for _ in range(max_calls-1)])
        mapping[bead, (bead):] = idx
        if bead < stats.n_beads - 1:
            mapping[(bead+1):, bead] = idx
    np.testing.assert_array_equal(stats.redundant_distance_mapping,
                                  mapping)
