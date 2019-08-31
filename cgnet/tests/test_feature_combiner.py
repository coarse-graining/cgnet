# Authors: Nick Charron

import numpy as np
import torch
import torch.nn as nn
from cgnet.feature import (FeatureCombiner, GeometryFeature, GeometryStatistics,
                           LinearLayer, SchnetFeature, CGBeadEmbedding)
from cgnet.network import (CGnet, ForceLoss, HarmonicLayer, ZscoreLayer)

# First, we construct random linear protein data
n_beads = np.random.randint(5, high=10)
n_frames = np.random.randint(10, high=30)
coords_numpy = np.random.randn(n_frames, n_beads, 3)
coords_torch = torch.tensor(coords_numpy, requires_grad=True).float()

# Next, we recover the statistics of random protein data and we produce the
# means, constants, and callback indices of bonds for a HarmonicLayer prior
geom_stats = GeometryStatistics(coords_numpy)
bonds_list, _ = geom_stats.get_prior_statistics('Bonds', as_list=True)
bonds_idx = geom_stats.return_indices('Bonds')
# Here we use the bond statistics to create a HarmonicLayer
bond_potential = HarmonicLayer(bonds_idx, bonds_list)

# Next, we produce the zscore statistics and create a ZscoreLayer
zscores, _ = geom_stats.get_zscore_array()
zscore_layer = ZscoreLayer(zscores)

# Next, we create a GeometryFeature layer for the subsequent tests
geometry_feature = GeometryFeature(n_beads=n_beads)

# Finally, we create random schnet init varaibles for the subsequent tests
feature_size = np.random.randint(5, high=10)  # random feature size
n_embeddings = np.random.randint(3, high=5)  # random embedding number
embedding_dim = feature_size  # embedding property size
n_interaction_blocks = np.random.randint(1, 3)  # random number of interactions
neighbor_cutoff = np.random.uniform(0, 1)  # random neighbor cutoff
# random embedding property
embedding_property = torch.randint(low=0, high=n_embeddings,
                                   size=(n_frames, n_beads))
embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings,
                                  embedding_dim=embedding_dim)


def test_combiner_geometry_feature():
    # Tests FeatureCombiner for just single GeometryFeature
    # First, we instantiate a FeatureCombiner
    layer_list = [geometry_feature]
    feature_combiner = FeatureCombiner(layer_list, save_geometry=False)

    # If there is simply a GeometryFeature, then feature_combiner.forward()
    # should return feature_ouput, geometry_output, with geometry_features
    # equal to None
    feature_output, geometry_output = feature_combiner(coords_torch)
    assert feature_combiner.transforms == [None]
    np.testing.assert_equal(list(feature_output.size()), list((n_frames,
                            len(geom_stats.master_description_tuples))))
    assert geometry_output is None


def test_combiner_schnet_feature():
    # Tests FeatureCombiner for just single SchnetFeature
    # First, we instantiate a FeatureCombiner with a SchnetFeature
    # That is capable of calculating pairwise distances (calculate_geometry
    # is True)
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=n_interaction_blocks,
                                   calculate_geometry=True,
                                   n_beads=n_beads,
                                   neighbor_cutoff=neighbor_cutoff)
    layer_list = [schnet_feature]
    feature_combiner = FeatureCombiner(layer_list)

    # If there is just a single SchnetFeature, then feature_ouput should
    # of shape [n_frames, n_beads, n_features] and geometry_features should
    # be None
    feature_output, geometry_output = feature_combiner(coords_torch,
                                      embedding_property=embedding_property)
    np.testing.assert_array_equal(feature_output.size(), (n_frames, n_beads,
                                                          feature_size))
    assert geometry_output is None


def test_combiner_zscore():
    # Tests FeatureCombiner for proper forwarding with GeometryFeature,
    # ZscoreLayer, and SchnetFeature. First, we make our FeatureCombiner
    layer_list = [geometry_feature, zscore_layer]
    feature_combiner = FeatureCombiner(layer_list)

    # If there is simply a GeometryFeature, then feature_combiner.forward()
    # should return feature_ouput, geometry_output, with geometry_features
    # equal to None
    feature_output, geometry_output = feature_combiner(coords_torch)
    # Both transfroms should be None
    assert feature_combiner.transforms == [None, None]
    np.testing.assert_equal(list(feature_output.size()), list((n_frames,
                            len(geom_stats.master_description_tuples))))
    np.testing.assert_equal(list(geometry_output.size()), list((n_frames,
                            len(geom_stats.master_description_tuples))))


def test_combiner_priors():
    # Test the combination of GeometryFeature, priors
    # within a CGnet instance. First, we create our FeatureCombiner
    layer_list = [geometry_feature, zscore_layer]
    feature_combiner = FeatureCombiner(layer_list)

    # Next, we create CGnet and use the bond_potential prior and
    # feature_combiner. We use a simple, random, four-layer hidden architecutre
    # for the terminal fully-connected layers
    width = np.random.randint(5, high=10)  # random fully-connected width
    arch = LinearLayer(len(geom_stats.master_description_tuples),
                       width, activation=nn.Tanh())
    for i in range(2):
        arch += LinearLayer(width, width, activation=nn.Tanh())
    arch += LinearLayer(width, 1, activation=None)
    model = CGnet(arch, ForceLoss(), feature=feature_combiner,
                  priors=[bond_potential])

    # Next, we forward the random protein data through the model
    # and assert the output has the correct shape
    energy, forces = model.forward(coords_torch)
    np.testing.assert_array_equal(energy.size(), (n_frames, 1))
    np.testing.assert_array_equal(forces.size(), (n_frames, n_beads, 3))

    # To test the priors, we compare to a CGnet formed with just
    # the tradiational feature=GeometryFeature init
    arch = [zscore_layer] + arch
    model_2 = CGnet(arch, ForceLoss(), feature=geometry_feature,
                    priors=[bond_potential])
    energy_2, forces_2 = model_2.forward(coords_torch)
    np.testing.assert_array_equal(energy.detach().numpy(),
                                  energy_2.detach().numpy())
    np.testing.assert_array_equal(forces.detach().numpy(),
                                  forces_2.detach().numpy())


def test_combiner_full():
    # Test the combination of GeometryFeature, SchnetFeature,
    # amd priors in a CGnet class
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=n_interaction_blocks,
                                   calculate_geometry=False,
                                   n_beads=n_beads,
                                   neighbor_cutoff=neighbor_cutoff)

    layer_list = [geometry_feature, zscore_layer, schnet_feature]
    # grab distance indices
    dist_idx = geom_stats.return_indices('Distances')
    feature_combiner = FeatureCombiner(layer_list, distance_indices=dist_idx)

    # Next, we create CGnet and use the bond_potential prior and
    # feature_combiner. We use a simple, random, four-layer hidden architecutre
    # for the terminal fully-connected layers
    width = np.random.randint(5, high=10)  # random fully-connected width
    arch = LinearLayer(feature_size,
                       width, activation=nn.Tanh())
    for i in range(2):
        arch += LinearLayer(width, width, activation=nn.Tanh())
    arch += LinearLayer(width, 1, activation=None)
    model = CGnet(arch, ForceLoss(), feature=feature_combiner,
                  priors=[bond_potential])

    # Next, we forward the random protein data through the model
    energy, forces = model.forward(coords_torch,
                                   embedding_property=embedding_property)

    # Ensure CGnet output has the correct size
    np.testing.assert_array_equal(energy.size(), (n_frames, 1))
    np.testing.assert_array_equal(forces.size(), (n_frames, n_beads, 3))
