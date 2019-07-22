# Author: Brooke Husic

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import ProteinBackboneFeature
from cgnet.feature import ProteinBackboneStatistics

frames = np.random.randint(1, 10)
beads = np.random.randint(8, 20)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)

f = ProteinBackboneFeature()
out = f.forward(xt)
stats = ProteinBackboneStatistics(xt)

backbone_inds = [i for i in range(beads) if i % 2 == 0]
xt_bb_only = xt[:,backbone_inds]


def test_manual_backbone_calculations():
    # Make sure angle statistics work for manually specified backbone
    stats_bb_inds = ProteinBackboneStatistics(xt, backbone_inds=backbone_inds)
    stats_bb_only = ProteinBackboneStatistics(xt_bb_only)

    np.testing.assert_allclose(stats_bb_inds.backbone_angles,
                               stats_bb_only.backbone_angles)

    np.testing.assert_allclose(stats_bb_inds.backbone_dihedral_cosines,
                               stats_bb_only.backbone_dihedral_cosines)

    np.testing.assert_allclose(stats_bb_inds.backbone_dihedral_sines,
                               stats_bb_only.backbone_dihedral_sines)

def test_manual_backbone_descriptions():
    # Make sure angle statistics work for manually specified backbone
    stats_bb_inds = ProteinBackboneStatistics(xt, backbone_inds=backbone_inds)
    stats_bb_only = ProteinBackboneStatistics(xt_bb_only)

    bb_ind_angle_descs = [(backbone_inds[i], backbone_inds[i+1], backbone_inds[i+2])
                          for i in range(len(backbone_inds)-2)]
    bb_only_angle_descs = [(i, i+1, i+2) for i in range(len(backbone_inds)-2)]

    bb_ind_dihed_descs = [(backbone_inds[i], backbone_inds[i+1],
                           backbone_inds[i+2], backbone_inds[i+3])
                           for i in range(len(backbone_inds)-3)]
    bb_only_dihed_descs = [(i, i+1, i+2, i+3) for i in range(len(backbone_inds)-3)]

    np.testing.assert_array_equal(stats_bb_inds.descriptions['Angles'],
                                  bb_ind_angle_descs)
    np.testing.assert_array_equal(stats_bb_only.descriptions['Angles'],
                                  bb_only_angle_descs)

    np.testing.assert_array_equal(stats_bb_inds.descriptions['Dihedral_cosines'],
                                  bb_ind_dihed_descs)
    np.testing.assert_array_equal(stats_bb_only.descriptions['Dihedral_cosines'],
                                  bb_only_dihed_descs)

    np.testing.assert_array_equal(stats_bb_inds.descriptions['Dihedral_sines'],
                                  bb_ind_dihed_descs)
    np.testing.assert_array_equal(stats_bb_only.descriptions['Dihedral_sines'],
                                  bb_only_dihed_descs)


def test_backbone_distance_statistics():
    # Make sure distance statistics are consistent with numpy

    feature_dist_mean = np.mean(f.distances.numpy(), axis=0)
    feature_dist_std = np.std(f.distances.numpy(), axis=0)

    np.testing.assert_allclose(feature_dist_mean,
                               stats.stats_dict['Distances']['mean'],
                               rtol=1e-4)
    np.testing.assert_allclose(feature_dist_std,
                               stats.stats_dict['Distances']['std'],
                               rtol=1e-4)


def test_backbone_angle_statistics():
    # Make sure angle statistics are consistent with numpy

    feature_angle_mean = np.mean(f.angles.numpy(), axis=0)
    feature_angle_std = np.std(f.angles.numpy(), axis=0)

    np.testing.assert_allclose(feature_angle_mean,
                               stats.stats_dict['Angles']['mean'], rtol=1e-5)
    np.testing.assert_allclose(feature_angle_std,
                               stats.stats_dict['Angles']['std'], rtol=1e-5)


def test_backbone_dihedral_statistics():
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
