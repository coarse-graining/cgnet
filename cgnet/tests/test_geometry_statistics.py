# Author: Brooke Husic
# Contributors : Nick Charron

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryFeature, GeometryStatistics

# The following sets up our pseud-simulation data

# Number of frames
frames = np.random.randint(1, 10)

# Number of coarse-grained beads. We need at least 8 so we can do
# dihedrals in the backbone tests (where every other atom is designated
# as a backbone atom)
beads = np.random.randint(8, 20)

# Number of dimensions; for now geometry only handles 3
dims = 3

# Create a pseudo simulation dataset
data = np.random.randn(frames, beads, dims).astype(np.float64)
data_tensor = torch.Tensor(data).double()

geom_feature = GeometryFeature(feature_tuples='all_backbone',
                                 n_beads=beads)
_ = geom_feature.forward(data_tensor)

stats = GeometryStatistics(data_tensor, backbone_inds='all',
                           get_all_distances=True,
                           get_backbone_angles=True,
                           get_backbone_dihedrals=True,
                           get_redundant_distance_mapping=True)


def test_feature_tuples():
    # Tests to see if the feature_tuples attribute is assembled correctly
    unique_tuples = []

    for desc in stats.order:  # for each type of feature
        sub_list = stats.descriptions[desc]  # list the feature tuples
        for bead_tuple in sub_list:
            if bead_tuple not in unique_tuples:
                unique_tuples.append(bead_tuple)
    np.testing.assert_array_equal(unique_tuples, stats.feature_tuples)


def test_custom_feature_shapes():
    # Tests whether statistics object has the right number of distances,
    # angles, and dihedrals when custom features are given

    # First generate the starts of features to be used for distances,
    # angles, and dihedrals
    custom_starts = np.unique([np.random.randint(beads - 4) for _ in range(5)])
    custom_distance_pairs = [(i, i+2) for i in custom_starts]
    custom_angle_triples = [(i, i+1, i+2) for i in custom_starts]
    custom_dihed_quads = [(i, i+1, i+2, i+3) for i in custom_starts]

    custom_features = (custom_distance_pairs +
                       custom_angle_triples + custom_dihed_quads)

    custom_stats = GeometryStatistics(data_tensor, custom_features)

    # We just want to make sure that no other features (like backbone)
    # showed up
    assert len(custom_starts) == len(custom_stats._distance_pairs)
    assert len(custom_starts) == len(custom_stats._angle_trips)
    assert len(custom_starts) == len(custom_stats._dihedral_quads)


def test_custom_feature_consistency_with_backbone():
    # Tests whether manually specifying backbone indices gives the same result
    # as automatically calculating backbone features

    backbone_angles = [(i, i+1, i+2) for i in range(beads - 2)]
    backbone_diheds = [(i, i+1, i+2, i+3) for i in range(beads - 3)]
    custom_features = backbone_angles + backbone_diheds

    backbone_stats = GeometryStatistics(data_tensor, backbone_inds='all',
                                        get_backbone_angles=True,
                                        get_backbone_dihedrals=True)
    backbone_stats_dict = backbone_stats.get_prior_statistics()

    custom_stats = GeometryStatistics(data_tensor, custom_features)
    custom_stats_dict = custom_stats.get_prior_statistics()

    np.testing.assert_array_equal(backbone_stats_dict, custom_stats_dict)


def test_manual_backbone_calculations():
    # Make sure backbone distance, angle, and dihedral statistics work
    # for manually specified backbone

    # Arbitrarily specify backbone indices
    # The test should be robust to changing this
    backbone_inds = [i for i in range(beads) if i % 2 == 0]

    # Create a backbone-only coordinate data tensor
    data_tensor_bb_only = data_tensor[:, backbone_inds]

    stats_bb_inds = GeometryStatistics(data_tensor,
                                       backbone_inds=backbone_inds,
                                       get_all_distances=True,
                                       get_backbone_angles=True,
                                       get_backbone_dihedrals=True)
    stats_bb_only = GeometryStatistics(data_tensor_bb_only,
                                       backbone_inds='all',
                                       get_all_distances=True,
                                       get_backbone_angles=True,
                                       get_backbone_dihedrals=True)

    # Distances will be different because there are a different number
    # of beads in each dataset, but the bonds (which default to the adjacent
    # backbone beads unless bond_pairs are specified) will be the same
    # in each case, so we use return_indices to get only the "bond" distances
    # for testing
    stats_bb_inds_bond_dists = stats_bb_inds.distances[:,
                        stats_bb_inds.return_indices(stats_bb_inds.bond_pairs)]
    stats_bb_only_bond_dists = stats_bb_only.distances[:,
                        stats_bb_only.return_indices(stats_bb_only.bond_pairs)]

    np.testing.assert_allclose(stats_bb_inds_bond_dists,
                               stats_bb_only_bond_dists)

    # Angles and dihedrals calculate the backbone only by default, so we don't
    # need to process these first
    np.testing.assert_allclose(stats_bb_inds.angles,
                               stats_bb_only.angles)

    np.testing.assert_allclose(stats_bb_inds.dihedral_cosines,
                               stats_bb_only.dihedral_cosines)

    np.testing.assert_allclose(stats_bb_inds.dihedral_sines,
                               stats_bb_only.dihedral_sines)


def test_manual_backbone_descriptions():
    # Make sure backbone distance, angle, and dihedral descriptions work
    # for manually specified backbone

    # Arbitrarily specify backbone indices
    # The test should be robust to changing this
    backbone_inds = [i for i in range(beads) if i % 2 == 0]

    # Create a backbone-only coordinate data tensor
    data_tensor_bb_only = data_tensor[:, backbone_inds]

    stats_bb_inds = GeometryStatistics(data_tensor,
                                       backbone_inds=backbone_inds,
                                       get_all_distances=True,
                                       get_backbone_angles=True,
                                       get_backbone_dihedrals=True)
    stats_bb_only = GeometryStatistics(data_tensor_bb_only,
                                       backbone_inds='all',
                                       get_all_distances=True,
                                       get_backbone_angles=True,
                                       get_backbone_dihedrals=True)

    # Manually specify what all the descriptions should be
    bb_inds_bond_descs = [(backbone_inds[i], backbone_inds[i+1])
                          for i in range(len(backbone_inds)-1)]
    bb_only_bond_descs = [(i, i+1) for i in range(len(backbone_inds)-1)]

    bb_ind_angle_descs = [(backbone_inds[i], backbone_inds[i+1],
                           backbone_inds[i+2])
                          for i in range(len(backbone_inds)-2)]
    bb_only_angle_descs = [(i, i+1, i+2) for i in range(len(backbone_inds)-2)]

    bb_ind_dihed_descs = [(backbone_inds[i], backbone_inds[i+1],
                           backbone_inds[i+2], backbone_inds[i+3])
                          for i in range(len(backbone_inds)-3)]
    bb_only_dihed_descs = [(i, i+1, i+2, i+3)
                           for i in range(len(backbone_inds)-3)]

    np.testing.assert_array_equal(stats_bb_inds.bond_pairs,
                                  bb_inds_bond_descs)
    np.testing.assert_array_equal(stats_bb_only.bond_pairs,
                                  bb_only_bond_descs)

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


def test_backbone_means_and_stds():
    # Make sure distance, angle, and dihedral statistics are consistent with
    # numpy

    feature_dist_mean = np.mean(geom_feature.distances.numpy(), axis=0)
    feature_dist_std = np.std(geom_feature.distances.numpy(), axis=0)

    np.testing.assert_allclose(feature_dist_mean,
                               stats._stats_dict['Distances']['mean'],
                               rtol=1e-6)
    np.testing.assert_allclose(feature_dist_std,
                               stats._stats_dict['Distances']['std'],
                               rtol=1e-6)

    feature_angle_mean = np.mean(geom_feature.angles.numpy(), axis=0)
    feature_angle_std = np.std(geom_feature.angles.numpy(), axis=0)

    np.testing.assert_allclose(feature_angle_mean,
                               stats._stats_dict['Angles']['mean'], rtol=1e-5)
    np.testing.assert_allclose(feature_angle_std,
                               stats._stats_dict['Angles']['std'], rtol=1e-5)

    feature_dihed_cos_mean = np.mean(geom_feature.dihedral_cosines.numpy(),
                                     axis=0)
    feature_dihed_cos_std = np.std(geom_feature.dihedral_cosines.numpy(),
                                   axis=0)
    feature_dihed_sin_mean = np.mean(geom_feature.dihedral_sines.numpy(),
                                     axis=0)
    feature_dihed_sin_std = np.std(geom_feature.dihedral_sines.numpy(),
                                   axis=0)

    np.testing.assert_allclose(feature_dihed_cos_mean,
                               stats._stats_dict['Dihedral_cosines']['mean'],
                               rtol=1e-4)
    np.testing.assert_allclose(feature_dihed_cos_std,
                               stats._stats_dict['Dihedral_cosines']['std'],
                               rtol=1e-4)

    np.testing.assert_allclose(feature_dihed_sin_mean,
                               stats._stats_dict['Dihedral_sines']['mean'],
                               rtol=1e-4)
    np.testing.assert_allclose(feature_dihed_sin_std,
                               stats._stats_dict['Dihedral_sines']['std'],
                               rtol=1e-4)


def test_prior_statistics_shape_1():
    # Make sure the "flipped" prior statistics dict has the right structure

    # We want to arbitrarily choose among the get_* arguments of
    # GeometryStatistics, so we create a bool_list that has at minimum
    # one True entry, and shuffle it
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(data_tensor, backbone_inds='all',
                                get_all_distances=bool_list[0],
                                get_backbone_angles=bool_list[1],
                                get_backbone_dihedrals=bool_list[2])

    zscore_dict = stats_.get_prior_statistics(flip_dict=True)

    # The outer keys in the flipped dictionary are the feature tuples.
    # So we can calculate how many keys it should have by knowing how
    # many of each calculated feature there should be.
    n_keys = (bool_list[0]*beads*(beads-1)/2 + bool_list[1]*(beads-2)
              + bool_list[2]*2*(beads-3))

    assert len(zscore_dict) == n_keys


def test_prior_statistics_shape_2():
    # Make sure the prior statistics dict has the right structure

    # We want to arbitrarily choose among the get_* arguments of
    # GeometryStatistics, so we create a bool_list that has at minimum
    # one True entry, and shuffle it
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(data_tensor, backbone_inds='all',
                                get_all_distances=bool_list[0],
                                get_backbone_angles=bool_list[1],
                                get_backbone_dihedrals=bool_list[2])

    zscore_dict = stats_.get_prior_statistics(flip_dict=False)
    n_keys = (bool_list[0]*beads*(beads-1)/2 + bool_list[1]*(beads-2)
              + bool_list[2]*2*(beads-3))

    # The INNER keys in the flipped dictionary are the feature tuples.
    # So we can calculate how many keys each entry of zscore_dict has
    # by knowing how many of each calculated feature there should be.
    for k in zscore_dict.keys():
        assert len(zscore_dict[k]) == n_keys


def test_prior_statistics():
    # Make sure distance means and stds are returned correctly

    # Here we choose some random beads at which to start bonds and then
    # create bonds of random lengths for our random starts
    bond_starts = [np.random.randint(beads-4) for _ in range(4)]
    bond_starts = np.unique(bond_starts)
    custom_bond_pairs = [(bs, bs+np.random.randint(1, 5))
                         for bs in bond_starts]

    # We manually calculate the means and stds of the bond distances
    pair_means = []
    pair_stds = []
    for pair in sorted(custom_bond_pairs):
        pair_means.append(np.mean(np.linalg.norm(data[:, pair[1], :]
                                                 - data[:, pair[0], :], axis=1)))
        pair_stds.append(np.std(np.linalg.norm(data[:, pair[1], :]
                                               - data[:, pair[0], :], axis=1)))

    # We input our custom bonds into stats, and see if they match our
    # manual calculations
    stats_dict = stats.get_prior_statistics(custom_bond_pairs, tensor=False)
    np.testing.assert_allclose(pair_means, [stats_dict[k]['mean']
                                            for k in sorted(stats_dict.keys())],
                               rtol=1e-6)
    np.testing.assert_allclose(pair_stds, [stats_dict[k]['std']
                                           for k in sorted(stats_dict.keys())],
                               rtol=1e-5)


def test_prior_statistics_2():
    # Make sure that prior statistics shuffle correctly

    # Here we create a shuffled list of feature tuples
    all_possible_features = stats.master_description_tuples
    my_inds = np.arange(len(all_possible_features))
    np.random.shuffle(my_inds)
    cutoff = np.random.randint(1, len(my_inds))
    my_inds = my_inds[:cutoff]

    all_stats = stats.get_prior_statistics()
    some_stats = stats.get_prior_statistics([all_possible_features[i]
                                             for i in my_inds])

    # We make sure that the shuffling returns the correct statistics
    # by indexing all_corresponding_dicts by our shuffled feature tuples
    some_dicts = [some_stats[k] for k in some_stats.keys()]
    all_corresponding_dicts = [all_stats[k] for k in some_stats.keys()]

    np.testing.assert_array_equal(some_dicts, all_corresponding_dicts)


def test_return_indices_shape_1():
    # Test proper retrieval of feature indices for sizes

    # We want to arbitrarily choose among the get_* arguments of
    # GeometryStatistics, so we create a bool_list that has at minimum
    # one True entry, and shuffle it
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(data_tensor, backbone_inds='all',
                                get_all_distances=bool_list[0],
                                get_backbone_angles=bool_list[1],
                                get_backbone_dihedrals=bool_list[2])

    # We know how many of each feature there should be, so we manually
    # calculate those numbers and compare them to what we get out.
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

    # We want to arbitrarily choose among the get_* arguments of
    # GeometryStatistics, so we create a bool_list that has at minimum
    # one True entry, and shuffle it
    bool_list = [True] + [bool(np.random.randint(2)) for _ in range(2)]
    np.random.shuffle(bool_list)

    stats_ = GeometryStatistics(data_tensor, backbone_inds='all',
                                get_all_distances=bool_list[0],
                                get_backbone_angles=bool_list[1],
                                get_backbone_dihedrals=bool_list[2])

    num_dists = bool_list[0] * (beads) * (beads - 1) / 2
    num_angles = beads - 2
    num_diheds = beads - 3

    # Here we want to make sure that our indices are correct, and we check
    # this by manually incrementing the indices using knowledge of how many
    # of each feature there should be
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

    # Here we choose some random beads at which to start bonds and then
    # create bonds of random lengths for our random starts
    bond_starts = [np.random.randint(beads-4) for _ in range(4)]
    bond_starts = np.unique(bond_starts)
    custom_bond_pairs = [(bs, bs+np.random.randint(2, 5))
                         for bs in bond_starts]

    # We input our custom bond pairs and do not care whether adjacent
    # backbone bonds are counted or not
    stats_ = GeometryStatistics(data_tensor, bond_pairs=custom_bond_pairs,
                                backbone_inds='all', get_all_distances=True,
                                adjacent_backbone_bonds=bool(np.random.randint(2)))
    returned_bond_inds = stats_.return_indices('Bonds')

    # We get the bond pairs from 'Distances' using the indices we know are bonds
    bond_pairs = np.array(stats_.descriptions['Distances'])[returned_bond_inds]

    # We remove any backbone bonds that may be present, since our construction
    # of bonds above only included bonds separated by at least one other bead
    bond_pairs = [tuple(bp) for bp in bond_pairs if bp[1]-bp[0] > 1]

    np.testing.assert_array_equal(sorted(custom_bond_pairs),
                                  sorted(bond_pairs))


def test_return_indices_and_prior_stats():
    # Test passing random tuples return_indices for size only

    all_beads = np.arange(beads)

    # For distances, angles, and dihedrals, we create random lists
    # of adjacent pairs, triples, and quads, respectively, and check that
    # their indices are properly from both return_indices and the
    # prior_statistics_dict
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
    trips = np.random.choice(all_beads[:-2],
                             size=np.random.randint(1, high=beads-2),
                             replace=False)
    angle_trips = [(all_beads[i], all_beads[i+1], all_beads[i+2])
                   for i in trips]
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


def test_prior_stats_list():
    # Tests as_list=True option in get_prior_statistics()
    # Tests to see if the proper statistics are returned for proper keys

    # First we shuffle the feature tuple-integer index pairs
    features = stats.master_description_tuples
    indices = stats.return_indices(features)
    zipped = list(zip(features, indices))
    np.random.shuffle(zipped)
    features[:], indices[:] = zip(*zipped)
    random_indices = stats.return_indices(features)

    # Next, we get the statistics as a dictionary and as a list
    # We also grab the keys (tuples of bead integers)
    prior_stats_dict = stats.get_prior_statistics(features=features)
    prior_stats_list, prior_stats_keys = stats.get_prior_statistics(
        features=features,
        as_list=True)
    # Next, we test to see if the shuffled keys were retrieved successfully
    np.testing.assert_array_equal(list(prior_stats_keys),
                                  [stats.master_description_tuples[i]
                                   for i in random_indices])
    # Next, we check to see if the stats in the list are ordered properly
    # and therefore correspond to the proper key/tuple of beads
    for stat_dict, stat_key in zip(prior_stats_list, prior_stats_keys):
        assert stat_dict == prior_stats_dict[stat_key]


def test_zscore_array_equivalence_to_prior_stats():
    # Tests to make sure keys are preserved from get_prior_statistics()
    # to get_zscore_array()

    # First we shuffle the feature tuple-integer index pairs
    features = stats.master_description_tuples
    indices = stats.return_indices(features)
    zipped = list(zip(features, indices))
    np.random.shuffle(zipped)
    features[:], indices[:] = zip(*zipped)

    # Then we get the indices and zscore values for our shuffled features
    prior_stats_list, prior_keys = stats.get_prior_statistics(features=features,
                                                              as_list=True)
    zscore_array, zscore_keys = stats.get_zscore_array(features=features)

    # We test the equivalence of the zscore keys to the prior statistics keys
    np.testing.assert_array_equal(prior_keys, zscore_keys)

    # We test the equivalence of the zscore values to the prior statistics values
    prior_means = [prior_stats_list[i]['mean']
                   for i in range(len(prior_stats_list))]
    prior_stds = [prior_stats_list[i]['std']
                  for i in range(len(prior_stats_list))]
    np.testing.assert_array_equal(prior_means, zscore_array[0])
    np.testing.assert_array_equal(prior_stds, zscore_array[1])

    # Note that zscore values aren't tested manually here, just for adherence
    # to prior statistics.


def test_redundant_distance_mapping_shape():
    # Test to see if the redundant distance index matrix has the right shape

    # The shape should be n x n - 1 for n beads
    index_mapping = stats.redundant_distance_mapping
    assert index_mapping.shape == (beads, beads - 1)

    # Craete mock distance data as a long vector per frame
    dist = np.random.randn(frames, int((beads - 1) * (beads) / 2))

    # Reshape it to be n x n - 1
    redundant_dist = dist[:, index_mapping]

    # Test the shape
    assert redundant_dist.shape == (frames, beads, beads - 1)


def test_redundant_distance_mapping_vals():
    # Test to see if the redundant distance index matrix has correct values

    # Here, we form the redundant mapping matrix by using shifted sequences
    # of triangle numbers (generated the neighbor_sequence function)
    # based on the default ordering of pairwise distances in GeometryStatistics
    mapping = np.zeros((stats.n_beads, stats.n_beads - 1), dtype='uint8')
    for bead in range(stats.n_beads):
        # Given the current bead integer, the sequence of neighbor bead distance
        # indices are generated with the use of the yeild statement
        def neighbor_sequence(_bead, n_beads):
            n = _bead
            j = n_beads - 1
            while(True):
                yield n + j
                n = n + j
                j -= 1
        # The above generator should only be called (n_beads - 2) times
        # becasue there are (n_beads - 2) distances to assemble after the
        # the first call
        max_calls_to_generator = stats.n_beads - bead - 1
        generator = neighbor_sequence(bead, stats.n_beads)
        # Here, the row of the bead neighbor distance indices is assembled
        index = np.array([bead] + [next(generator)
                                   for _ in range(max_calls_to_generator-1)])
        # The above row is then inserted into the mapping matrix to fill the
        # columns that occur after the current bead cloumn index
        mapping[bead, (bead):] = index
        # The same pattern in the row of indices extends down column-wise
        if bead < stats.n_beads - 1:
            mapping[(bead+1):, bead] = index
        # Ultimately, we are left with an index maping matrix that is
        # nearly symmetric (though obviously not symmetric because it is not
        # a squre matrix)
    # We test the above form against the method provided by GeometryStatistics
    np.testing.assert_array_equal(stats.redundant_distance_mapping,
                                  mapping)
