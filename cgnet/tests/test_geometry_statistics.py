# Author: Brooke Husic
# Contributors : Nick Charron

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryFeature, GeometryStatistics

frames = np.random.randint(1, 10)
beads = np.random.randint(8, 20)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)

f = GeometryFeature(n_beads=beads)
out = f.forward(xt)

stats = GeometryStatistics(xt)

def test_manual_backbone_calculations():
    # Make sure angle statistics work for manually specified backbone
    backbone_inds = [i for i in range(beads) if i % 2 == 0]
    xt_bb_only = xt[:, backbone_inds]

    stats_bb_inds = GeometryStatistics(xt, backbone_inds=backbone_inds)
    stats_bb_only = GeometryStatistics(xt_bb_only)

    np.testing.assert_allclose(stats_bb_inds.angles,
                               stats_bb_only.angles)

    np.testing.assert_allclose(stats_bb_inds.dihedral_cosines,
                               stats_bb_only.dihedral_cosines)

    np.testing.assert_allclose(stats_bb_inds.dihedral_sines,
                               stats_bb_only.dihedral_sines)


def test_manual_backbone_descriptions():
    # Make sure angle statistics work for manually specified backbone
    backbone_inds = [i for i in range(beads) if i % 2 == 0]
    xt_bb_only = xt[:, backbone_inds]

    stats_bb_inds = GeometryStatistics(xt, backbone_inds=backbone_inds)
    stats_bb_only = GeometryStatistics(xt_bb_only)

    bb_ind_angle_descs = [(backbone_inds[i], backbone_inds[i+1], backbone_inds[i+2])
                          for i in range(len(backbone_inds)-2)]
    bb_only_angle_descs = [(i, i+1, i+2) for i in range(len(backbone_inds)-2)]

    bb_ind_dihed_descs = [(backbone_inds[i], backbone_inds[i+1],
                           backbone_inds[i+2], backbone_inds[i+3])
                          for i in range(len(backbone_inds)-3)]
    bb_only_dihed_descs = [(i, i+1, i+2, i+3)
                           for i in range(len(backbone_inds)-3)]

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
                               stats._stats_dict['Distances']['mean'],
                               rtol=1e-4)
    np.testing.assert_allclose(feature_dist_std,
                               stats._stats_dict['Distances']['std'],
                               rtol=1e-4)


def test_backbone_angle_statistics():
    # Make sure angle statistics are consistent with numpy

    feature_angle_mean = np.mean(f.angles.numpy(), axis=0)
    feature_angle_std = np.std(f.angles.numpy(), axis=0)

    np.testing.assert_allclose(feature_angle_mean,
                               stats._stats_dict['Angles']['mean'], rtol=1e-5)
    np.testing.assert_allclose(feature_angle_std,
                               stats._stats_dict['Angles']['std'], rtol=1e-5)


def test_backbone_dihedral_statistics():
    # Make sure dihedral statistics are consistent with numpy

    feature_dihed_cos_mean = np.mean(f.dihedral_cosines.numpy(), axis=0)
    feature_dihed_cos_std = np.std(f.dihedral_cosines.numpy(), axis=0)
    feature_dihed_sin_mean = np.mean(f.dihedral_sines.numpy(), axis=0)
    feature_dihed_sin_std = np.std(f.dihedral_sines.numpy(), axis=0)

    np.testing.assert_allclose(feature_dihed_cos_mean,
                               stats._stats_dict['Dihedral_cosines']['mean'],
                               rtol=1e-6)
    np.testing.assert_allclose(feature_dihed_cos_std,
                               stats._stats_dict['Dihedral_cosines']['std'],
                               rtol=1e-6)

    np.testing.assert_allclose(feature_dihed_sin_mean,
                               stats._stats_dict['Dihedral_sines']['mean'],
                               rtol=1e-6)
    np.testing.assert_allclose(feature_dihed_sin_std,
                               stats._stats_dict['Dihedral_sines']['std'],
                               rtol=1e-6)


def test_prior_statistics_1():
    # Make sure the "flipped" prior statistics dict has the right structure
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(xt,
                               get_all_distances=bool_list[0],
                               get_backbone_angles=bool_list[1],
                               get_backbone_dihedrals=bool_list[2])

    zscore_dict = stats_.get_prior_statistics(flip_dict=True)
    n_keys = (bool_list[0]*beads*(beads-1)/2 + bool_list[1]*(beads-2)
              + bool_list[2]*2*(beads-3))

    assert len(zscore_dict) == n_keys


def test_prior_statistics_2():
    # Make sure the prior statistics dict has the right structure
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(xt,
                               get_all_distances=bool_list[0],
                               get_backbone_angles=bool_list[1],
                               get_backbone_dihedrals=bool_list[2])

    zscore_dict = stats_.get_prior_statistics(flip_dict=False)
    n_keys = (bool_list[0]*beads*(beads-1)/2 + bool_list[1]*(beads-2)
              + bool_list[2]*2*(beads-3))

    for k in zscore_dict.keys():
        assert len(zscore_dict[k]) == n_keys


def test_return_indices_1():
    # Test proper retrieval of feature indices for sizes
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(xt,
                               get_all_distances=bool_list[0],
                               get_backbone_angles=bool_list[1],
                               get_backbone_dihedrals=bool_list[2])

    if bool_list[0]:
        assert len(stats_.return_indices('Distances')) == (
            beads) * (beads - 1) / 2
        assert len(stats_.return_indices('Bonds')) == beads - 1
    if bool_list[1]:
        assert len(stats_.return_indices('Angles')) == beads - 2
    if bool_list[2]:
        assert len(stats_.return_indices('Dihedral_cosines')) == beads - 3
        assert len(stats_.return_indices('Dihedral_sines')) == beads - 3

    sum_feats = np.sum([len(stats_.descriptions[feat_name])
                        for feat_name in stats_.order])
    check_sum_feats = (bool_list[0] * (beads) * (beads - 1) / 2 +
                       bool_list[1] * (beads - 2) +
                       bool_list[2] * (beads - 3) * 2
                       )
    assert sum_feats == check_sum_feats


def test_return_indices_2():
    # Test proper retrieval of feature indices for specific indices
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(xt,
                               get_all_distances=bool_list[0],
                               get_backbone_angles=bool_list[1],
                               get_backbone_dihedrals=bool_list[2])

    num_dists = bool_list[0] * (beads) * (beads - 1) / 2
    num_angles = beads - 2
    num_diheds = beads - 3

    if bool_list[0]:
        np.testing.assert_array_equal(np.arange(0, num_dists),
                                      stats_.return_indices('Distances'))

        bond_ind_list = [ind for ind, pair in enumerate(
            stats.descriptions['Distances'])
            if pair[1] - pair[0] == 1]
        np.testing.assert_array_equal(bond_ind_list,
                                      stats_.return_indices('Bonds'))

    if bool_list[1]:
        angle_start = bool_list[0]*num_dists
        np.testing.assert_array_equal(np.arange(angle_start,
                                                num_angles + angle_start),
                                      stats_.return_indices('Angles'))

    if bool_list[2]:
        dihedral_cos_start = bool_list[0]*num_dists + bool_list[1]*num_angles
        np.testing.assert_array_equal(np.arange(dihedral_cos_start,
                                                num_diheds + dihedral_cos_start),
                                      stats_.return_indices('Dihedral_cosines'))

        dihedral_sin_start = dihedral_cos_start + num_diheds
        np.testing.assert_array_equal(np.arange(dihedral_sin_start,
                                                num_diheds + dihedral_sin_start),
                                      stats_.return_indices('Dihedral_sines'))

def test_return_indices_3():
    # Test retrival of custom bonds
    bond_starts = [np.random.randint(beads-4) for _ in range(4)]
    bond_starts = np.unique(bond_starts)
    custom_bond_pairs = [(bs, bs+np.random.randint(2,5)) for bs in bond_starts]

    stats_ = GeometryStatistics(xt, bond_pairs=custom_bond_pairs,
                                adjacent_backbone_bonds = bool(np.random.randint(2)))
    returned_bond_inds = stats_.return_indices('Bonds')
    bond_pairs = np.array(stats_.descriptions['Distances'])[returned_bond_inds]
    bond_pairs = [tuple(bp) for bp in bond_pairs if bp[1]-bp[0]>1]

    np.testing.assert_array_equal(sorted(custom_bond_pairs),
                                  sorted(bond_pairs))

def test_return_indices_4():
    # Test passing random tuples to return_indices method
    # distance pairs
    bead_list = np.arange(beads)
    sub_beads = np.random.randint(2, high=beads)
    pairs = np.sort(np.random.choice(bead_list, size=sub_beads, replace=False))
    distance_pairs = [(pairs[i], pairs[i+1]) for i in range(len(pairs) - 1)]
    dist_idx = stats.return_indices(distance_pairs)
    assert len(dist_idx) == len(distance_pairs)

    # angles
    num_triplets = np.random.randint(1, high=beads - 2)
    bases = np.arange(num_triplets)
    angles = [(bases[i], bases[i+1], bases[i+2]) for i in bases]
    angles_idx = stats.return_indices(angles)
    assert len(angles_idx) == len(angles)

    # dihedrals
    num_quads = np.random.randint(1, high=beads - 3)
    bases = np.arange(num_quads)
    dihedrals = [(bases[i], bases[i+1], bases[i+2]) for i in bases]
    diheds_idx = stats.return_indices(dihedrals)
    # both sin and cos are returned for dihedrals
    assert len(diheds_idx) == 2 * len(dihedals)

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
        def neighbor_sequence(bead, n_beads):
            n = bead
            j = n_beads - 1
            while(True):
                yield n + j
                n = n + j
                j -= 1
        max_calls_to_generator = stats.n_beads - bead - 1
        generator = neighbor_sequence(bead, stats.n_beads)
        index = np.array([bead] + [next(generator)
                                   for _ in range(max_calls_to_generator-1)])
        mapping[bead, (bead):] = index
        if bead < stats.n_beads - 1:
            mapping[(bead+1):, bead] = index
    np.testing.assert_array_equal(stats.redundant_distance_mapping,
                                  mapping)
