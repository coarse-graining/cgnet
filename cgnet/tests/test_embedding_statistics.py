# Author: Nick Charron 

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryStatistics, EmbeddingStatistics

# The following sets up our pseud-simulation data

# Number of frames
frames = np.random.randint(5, 10) * 2 #double this to avoid nonzero variances

# Number of coarse-grained beads. We need at least 8 so we can do
# dihedrals in the backbone tests (where every other atom is designated
# as a backbone atom)
beads = np.random.randint(8, 20)

# Number of dimensions; for now geometry only handles 3
dims = 3

# Create a pseudo simulation dataset, but double each frame to get nonzero variances
data = np.random.randn(frames, beads, dims)


# Create embeddings that vary from frame to frame to simulate multiple
# different molecules within the same dataset

embeddings = np.random.randint(low=1, high=10, size=(int(frames/2),beads)) #again,double this so
# we dont get nonzero variances.

embeddings = np.tile(embeddings, (2,1))

# random temperature
temperature = np.random.uniform(low=250,high=350)
KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184
beta = JPERKCAL/KBOLTZMANN/AVOGADRO/temperature

stats = EmbeddingStatistics(data, embeddings, backbone_inds='all',
                           get_all_distances=True,
                           get_backbone_angles=True,
                           get_backbone_dihedrals=True,
                           get_redundant_distance_mapping=True,
                           temperature=temperature)
beta = stats.beta

def test_stats_dictionary_generation():
    # Tests to make sure that the embedding-bead mapping is
    # carried out correctly in the _get_*() methods

    # Manual bead-mapped-embeddings

    manual_distance_embeddings = embeddings[:, stats._distance_pairs]
    manual_angle_embeddings = embeddings[:, stats._angle_trips]
    manual_dihedral_embeddings = embeddings[:, stats._dihedral_quads]

    # Assert equality between ststs attributes and manual calculations
    np.testing.assert_equal(manual_distance_embeddings, stats.distance_embeddings)
    np.testing.assert_equal(manual_angle_embeddings, stats.angle_embeddings)
    np.testing.assert_equal(manual_dihedral_embeddings, stats.dihedral_embeddings)

def test_stats_calculation():
    # Tests to see that the sefl._stats_dict is assembled properly given the embeddings

    manual_distance_embeddings = embeddings[:, stats._distance_pairs]
    manual_angle_embeddings = embeddings[:, stats._angle_trips]
    manual_dihedral_embeddings = embeddings[:, stats._dihedral_quads]

    manual_embeddings = [manual_distance_embeddings, manual_angle_embeddings, manual_dihedral_embeddings,
                         manual_dihedral_embeddings]
    feature_keys = ['Distances', 'Angles', 'Dihedral_sines', 'Dihedral_cosines']
    features = [stats.distances, stats.angles, stats.dihedral_sines, stats.dihedral_cosines]
    beads = [stats._distance_pairs, stats._angle_trips, stats._dihedral_quads, stats._dihedral_quads]

    manual_stats = {}

    for feature, feat_type, bead, manual_embedding in zip(features, feature_keys, beads, manual_embeddings):
        manual_stats[feat_type] = {}
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                embedding_key = tuple((*bead[j],*manual_embedding[i,j,:]))
                if embedding_key not in manual_stats[feat_type].keys():
                    manual_stats[feat_type][embedding_key] = {}
                    manual_stats[feat_type][embedding_key]['mean'] = 0
                    manual_stats[feat_type][embedding_key]['std'] = 0
                    manual_stats[feat_type][embedding_key]['var'] = 0
                    manual_stats[feat_type][embedding_key]['k'] = 0
                    manual_stats[feat_type][embedding_key]['observations'] = 0

                manual_stats[feat_type][embedding_key]['mean'] += feature[i,j]
                manual_stats[feat_type][embedding_key]['observations'] += 1
        #compute final means
        for embedding_key in manual_stats[feat_type].keys():
            manual_stats[feat_type][embedding_key]['mean'] /= manual_stats[feat_type][embedding_key]['observations']

        #variance calulcation - maybe update with Welford's algorithm if too slow
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                embedding_key = tuple((*bead[j],*manual_embedding[i,j,:]))
                manual_stats[feat_type][embedding_key]['var'] += (feature[i,j]
                    - manual_stats[feat_type][embedding_key]['mean'])**2
        #final vars , stds, and k       
        for embedding_key in manual_stats[feat_type].keys():
            manual_stats[feat_type][embedding_key]['var'] /= manual_stats[feat_type][embedding_key]['observations']
            manual_stats[feat_type][embedding_key]['std'] = np.sqrt(manual_stats[feat_type][embedding_key]['var'])
            manual_stats[feat_type][embedding_key]['k'] = 1.0/manual_stats[feat_type][embedding_key]['var']/beta

    # finally, we test the equality of the manual stats dictionary and self._stats_dict

    for key in manual_stats.keys():
        assert set(manual_stats.keys()) == set(stats._stats_dict.keys())
        for key1, key2 in zip(sorted(manual_stats[key].keys()), sorted(stats._stats_dict[key].keys())):
            assert stats._stats_dict[key][key2]['observations'] > 0
            assert manual_stats[key][key1]['mean'] == stats._stats_dict[key][key2]['mean']
            assert manual_stats[key][key1]['std'] == stats._stats_dict[key][key2]['std']
            assert manual_stats[key][key1]['var'] == stats._stats_dict[key][key2]['var']
            assert manual_stats[key][key1]['k'] == stats._stats_dict[key][key2]['k']
            assert manual_stats[key][key1]['observations'] == stats._stats_dict[key][key2]['observations']

def test_statistics_grabber():
    # Tests stats.get_prior_statistics (and by proxy, get_prior_statistics_helper()) to
    # make sure that statistics can be assembled for a random assortment of features
    n_distances = np.random.randint(0, high=len(stats._distance_pairs))
    n_angles = np.random.randint(0, high=len(stats._angle_trips))
    n_dihedral_sines = np.random.randint(0, high=len(stats._dihedral_quads))
    n_dihedral_cosines = np.random.randint(0, high=len(stats._dihedral_quads))

    distance_idx = np.random.choice(np.arange(0, len(stats._distance_pairs)), n_distances, replace=False)
    angle_idx = np.random.choice(np.arange(0, len(stats._angle_trips)), n_angles, replace=False)
    dihedral_sines_idx = np.random.choice(np.arange(0, len(stats._dihedral_quads)), n_dihedral_sines, replace=False)
    dihedral_cosines_idx = np.random.choice(np.arange(0, len(stats._dihedral_quads)), n_dihedral_cosines, replace=False)

    distance_features = [stats._distance_pairs[i] for i in distance_idx]
    angle_features = [stats._angle_trips[i] for i in angle_idx]
    dihedral_sine_features = [stats._dihedral_quads[i] for i in dihedral_sines_idx]
    dihedral_cosine_features = [stats._dihedral_quads[i] for i in dihedral_cosines_idx]

    distance_embeddings = stats.distance_embeddings[:, distance_idx, :]
    angle_embeddings = stats.angle_embeddings[:, angle_idx, :]
    dihedral_sine_embeddings = stats.dihedral_embeddings[:, dihedral_sines_idx, :]
    dihedral_cosine_embeddings = stats.dihedral_embeddings[:, dihedral_cosines_idx, :]

    dihedral_sine_features = [tuple((*feat, "sine")) for feat in dihedral_sine_features]
    dihedral_cosine_features = [tuple((*feat, "cosine")) for feat in dihedral_cosine_features]

    compound_distance_keys = [[tuple((*bead, *embed)) for bead, embed in zip(distance_features, distance_embeddings[j])]
                                 for j in range(distance_embeddings.shape[0])]
    compound_angle_keys = [[tuple((*bead, *embed)) for bead, embed in zip(angle_features, angle_embeddings[j])]
                               for j in range(angle_embeddings.shape[0])]
    compound_dihedral_sine_keys = [[tuple((*bead[:-1], *embed)) for bead, embed in zip(dihedral_sine_features, dihedral_sine_embeddings[j])]
                                      for j in range(dihedral_sine_embeddings.shape[0])]
    compound_dihedral_cosine_keys = [[tuple((*bead[:-1], *embed)) for bead, embed in zip(dihedral_cosine_features, dihedral_cosine_embeddings[j])]
                                       for j in range(dihedral_cosine_embeddings.shape[0])]

    compound_distance_keys = [item for sublist in compound_distance_keys for item in sublist]
    compound_angle_keys = [item for sublist in compound_angle_keys for item in sublist]
    compound_dihedral_sine_keys = [item for sublist in compound_dihedral_sine_keys for item in sublist]
    compound_dihedral_cosine_keys = [item for sublist in compound_dihedral_cosine_keys for item in sublist]

    total_features = distance_features + angle_features + dihedral_sine_features + dihedral_cosine_features
    np.random.shuffle(total_features)

    # Here, we produce the grabbed features and check to see if the resulting dictionary 
    # contains all of the requested feature types and tuples

    grabbed_feature_stats = stats.get_prior_statistics(total_features)
    #print(compound_distance_keys)
    stats_keys = ['mean', 'var', 'std', 'k', 'observations']
    if n_distances > 0:
        assert "Distances" in grabbed_feature_stats.keys()
        assert set(compound_distance_keys) == set(grabbed_feature_stats["Distances"].keys())
        for feat in compound_distance_keys:
            for stat_key in stats_keys:
                assert grabbed_feature_stats["Distances"][feat][stat_key] == torch.tensor(
                    stats._stats_dict["Distances"][feat][stat_key])

    if n_angles > 0:
        assert "Angles" in grabbed_feature_stats.keys()
        assert set(compound_angle_keys) == set(grabbed_feature_stats["Angles"].keys())
        for feat in compound_angle_keys:
            for stat_key in stats_keys:
                assert grabbed_feature_stats["Angles"][feat][stat_key] == torch.tensor(
                    stats._stats_dict["Angles"][feat][stat_key])

    if n_dihedral_sines > 0:
        assert "Dihedral_sines" in grabbed_feature_stats.keys()
        print(compound_dihedral_sine_keys)
        print(grabbed_feature_stats["Dihedral_sines"].keys())
        assert set(compound_dihedral_sine_keys) == set(grabbed_feature_stats["Dihedral_sines"].keys())
        for feat in compound_dihedral_sine_keys:
            for stat_key in stats_keys:
                assert grabbed_feature_stats["Dihedral_sines"][feat][stat_key] == torch.tensor(
                    stats._stats_dict["Dihedral_sines"][feat][stat_key])

    if n_dihedral_cosines > 0:
        assert "Dihedral_cosines" in grabbed_feature_stats.keys()
        assert set(compound_dihedral_cosine_keys) == set(grabbed_feature_stats["Dihedral_cosines"].keys())
        for feat in compound_dihedral_cosine_keys:
            for stat_key in stats_keys:
                assert grabbed_feature_stats["Dihedral_cosines"][feat][stat_key] == torch.tensor(
                    stats._stats_dict["Dihedral_cosines"][feat][stat_key])


