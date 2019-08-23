# Authors: Nick Charron

import numpy as np
import torch
import torch.nn as nn
from cgnet.feature import (FeatureCombiner, GeometryFeature, GeometryStatistics)
from cgnet.network import (CGnet, ForceLoss, HarmonicLayer, ZscoreLayer)

# First, we construct random linear protein data
n_beads = np.random.randint(5, high=10)
n_frames = np.random.randint(10, high=30)
coords_numpy = np.random.randn(n_frames, n_beads, 3)
coords_torch = torch.tensor(coords_numpy, requires_grad=True).float()

# Next, we recover the statistics of random protein data and we produce the
# means, constants, and callback indices of bonds for a HarmonicLayer prior
stats = GeometryStatistics(coords_numpy)
bonds_list, _ = stats.get_prior_statistics('Bonds', as_list=True)
bonds_idx = stats.return_indices('Bonds')

# Here we use the bond statistics to create a HarmonicLayer
bond_potential = HarmonicLayer(bonds_idx, bonds_list)

# Next, we produce the zscore statistics and create a ZscoreLayer
zscores, _ = stats.get_zscore_array()
zscore_layer = ZscoreLayer(zscores)

# Next, we create a GeometryFeature layer and a SchNet feature
# layer for the subsequent tests
geometry_feature = GeometryFeature(feature_tuples=stats.feature_tuples)
#schnet_feature = SchnetFeature()
"""
def test_combiner_assembly():
    # Test combiner initialization


def test_cgnet_feature():
    # Test use of FeatureCombiner as a CGnet feature
"""
def test_combiner_geometry():
    # Tests FeatureCombiner for just single GeometryFeature
    # First, we instantiate a FeatureCombiner
    layer_list = [geometry_feature]
    feature_combiner = FeatureCombiner(layer_list)

    # If there is simply a GeometryFeature, then feature_combiner.forward()
    # should return feature_ouput, geometry_output, with geometry_features
    # equal to None
    feature_output, geometry_output = feature_combiner(coords_torch)
    assert feature_combiner.transforms == [None]
    np.testing.assert_equal(list(feature_output.size()), list((n_frames,
                        len(stats.master_description_tuples))))
    assert geometry_output is None

"""
def test_combiner_schnet():
    # Tests FeatureCombiner for just single SchnetFeature
    # First, we instantiate a FeatureCombiner
    layer_list = [schnet_feature]
    feature_combiner = FeatureCombiner(layer_list)

    # If there is just a single SchnetFeature, then feature_ouput should
    # of shape [n_frames, n_beads, n_features] and geometry_features should
    # be None
    feature_output, geometry_output = feature_combiner(coords_torch)
    np.testing.assert_equal(feature_ouput.size(), (n_frames,
                        len(stats._master_description_tuples))
    assert geometry_ouput is None
"""
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
                        len(stats.master_description_tuples))))
    np.testing.assert_equal(list(geometry_output.size()), list((n_frames,
                        len(stats.master_description_tuples))))


def test_combiner_priors():
    # Test the combination of GeometryFeature, priors
    # within a CGnet instance. First, we create our FeatureCombiner
    layer_list = [geometry_feature, zscore_layer]
    feature_combiner = FeatureCombiner(layer_list,
                                       get_redundant_distance_mapping=True)

    # Next, we create CGnet and use the bond_potential prior and
    # feature_combiner. We use a simple, random, four-layer hidden architecutre
    # for the terminal fully-connected layers
    width = np.random.randint(5, high=10) # random fully-connected width
    arch = LinearLayer(coords_torch.size()[1], width, activation=nn.Tanh())
    arch += [LinearLayer(width, width, activation=nn.Tanh()) for _ in range(2)]
    arch += LinearLayer(width, 1, acitvation=None)
    model = CGnet(arch, ForceLoss(), feature=feature_combiner,
                  priors=[bond_potential])

    # Next, we forward the random protein data through the model
    energy, forces = model.forward(coords_torch)

    # To test the priors, we compare to a CGnet formed with just  

