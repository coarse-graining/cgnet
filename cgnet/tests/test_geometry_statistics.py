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

def test_feature_tuples():
    # Tests to see if the feature_tuples attribute is assembled correctly
	unique_tuples = []
	for desc in stats.order:
		sub_list = stats.descriptions[desc]
		for bead_tuple in sub_list:
			if bead_tuple not in unique_tuples:
				unique_tuples.append(bead_tuple)
	assert unique_tuples == stats.feature_tuples

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


def test_prior_statistics_shape_1():
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


def test_prior_statistics_shape_2():
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


def test_prior_statistics():
    # Make sure distance means and stds are returned correctly

    bond_starts = [np.random.randint(beads-4) for _ in range(4)]
    bond_starts = np.unique(bond_starts)
    custom_bond_pairs = [(bs, bs+np.random.randint(1,5)) for bs in bond_starts]
    pair_means = []
    pair_stds = []
    for pair in sorted(custom_bond_pairs):
        pair_means.append(np.mean(np.linalg.norm(x[:,pair[1],:]
                                        - x[:,pair[0],:], axis=1)))
        pair_stds.append(np.std(np.linalg.norm(x[:,pair[1],:]
                                        - x[:,pair[0],:], axis=1)))
    stats_dict = stats.get_prior_statistics(custom_bond_pairs, tensor=False)
    np.testing.assert_allclose(pair_means, [stats_dict[k]['mean']
                               for k in sorted(stats_dict.keys())],
                               rtol=1e-6)
    np.testing.assert_allclose(pair_stds, [stats_dict[k]['std']
                               for k in sorted(stats_dict.keys())],
                               rtol=1e-5)


def test_prior_statistics_2():
    # Make sure that prior statistics shuffle correctly
    all_possible_features = stats.master_description_tuples
    my_inds = np.arange(len(all_possible_features))
    np.random.shuffle(my_inds)

    cutoff = np.random.randint(1, len(my_inds))
    my_inds = my_inds[:cutoff]

    all_stats = stats.get_prior_statistics()
    some_stats = stats.get_prior_statistics([all_possible_features[i]
                                                for i in my_inds])

    some_keys = [some_stats[k] for k in some_stats.keys()]
    all_corresponding_keys = [all_stats[k] for k in some_stats.keys()]

    np.testing.assert_array_equal(some_keys, all_corresponding_keys)


def test_return_indices_shape_1():
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


def test_return_indices_1():
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

def test_return_indices_2():
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

def test_return_indices_and_prior_stats():
    # Test passing random tuples return_indices for size only

    all_beads = np.arange(beads)

    pairs = np.random.choice(all_beads[:-1],
                             size=np.random.randint(2, high=beads-1),
                             replace=False)
    distance_pairs = [(all_beads[i], all_beads[i+1]) for i in pairs]
    dist_idx = stats.return_indices(distance_pairs)
    assert len(dist_idx) == len(distance_pairs)
    np.testing.assert_array_equal(distance_pairs,
                                  list(stats.get_prior_statistics(
                                    distance_pairs).keys()))

    # angles
    angle_start_list = np.arange(beads-2)
    trips = np.random.choice(all_beads[:-2],
                             size=np.random.randint(1, high=beads-2),
                             replace=False)
    angle_trips = [(all_beads[i], all_beads[i+1], all_beads[i+2]) for i in trips]
    angle_idx = stats.return_indices(angle_trips)
    assert len(angle_idx) == len(angle_trips)
    np.testing.assert_array_equal(angle_trips,
                                  list(stats.get_prior_statistics(
                                    angle_trips).keys()))

    # dihedrals
    quads = np.random.choice(all_beads[:-3],
                             size=np.random.randint(1, high=beads-3),
                             replace=False)
    dihed_quads = [(all_beads[i], all_beads[i+1],
                    all_beads[i+2], all_beads[i+3], 'cos') for i in quads]
    dihed_idx = stats.return_indices(dihed_quads)
    assert len(dihed_idx) == len(dihed_quads)
    np.testing.assert_array_equal(dihed_quads,
                                  list(stats.get_prior_statistics(
                                    dihed_quads).keys()))


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
