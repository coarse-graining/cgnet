# Author: Dominik Lemm

import numpy as np
import torch

from cgnet.feature import (ContinuousFilterConvolution, InteractionBlock,
                           GeometryStatistics, GeometryFeature,
                           RadialBasisFunction, SchnetBlock)

# Random protein
num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)
coords = np.random.randn(num_examples, num_beads, 3)

stats = GeometryStatistics(coords, get_redundant_distance_mapping=True)
feat_layer = GeometryFeature(feature_tuples=stats.feature_tuples)
output = feat_layer(torch.tensor(coords, requires_grad=True).float())
dist_idx = stats.return_indices('Distances')
distances = output[:, stats.redundant_distance_mapping]

# Random feature sizes
num_gaussians = np.random.randint(5, 10)
num_feats = np.random.randint(4, 16)
variance = np.random.rand() + 1.0
num_filters = num_feats

# Create test input features, radial basis function output and neighbor list
test_features = torch.randn((num_examples, num_beads, num_feats))
test_cfconv_features = torch.randn((num_examples, num_beads, num_filters))


rbf_layer = RadialBasisFunction(stats.redundant_distance_mapping,
                                num_gaussians=num_gaussians, variance=variance)

test_nbh = np.tile(np.arange(num_beads), (num_beads, 1))
inverse_identity = np.eye(num_beads, num_beads) != 1
test_nbh = torch.from_numpy(test_nbh[inverse_identity].reshape(num_beads,
                                                                  num_beads - 1))

print(test_nbh)
def test_continuous_convolution():
    # Comparison of the ContinuousFilterConvolution with a manual numpy calculation

    # Calculate continuous convolution output with the created layer
    cfconv = ContinuousFilterConvolution(test_nbh,
                                         num_gaussians=num_gaussians,
                                         num_filters=num_filters)
    rbf_out = rbf_layer(distances)
    cfconv_layer_out = cfconv.forward(test_cfconv_features, rbf_out).detach()

    # Calculate convolution manually
    test_conv_filter = cfconv.filter_generator(rbf_out).detach().numpy()

    num_neighbors = num_beads - 1
    test_nbh_np = test_nbh.numpy()
    test_feat_np = test_cfconv_features.numpy()

    # Gather the features into the respective places in the neighbor list
    neighbor_list = test_nbh_np.reshape(-1, num_beads * num_neighbors, 1)
    neighbor_list = neighbor_list.repeat(num_filters, axis=2)
    neighbor_features = np.take_along_axis(test_feat_np, neighbor_list, axis=1)
    neighbor_features = neighbor_features.reshape(num_examples, num_beads,
                                                  num_neighbors, -1)

    # element-wise multiplication and pooling
    conv_features = neighbor_features * test_conv_filter
    cfconv_manual_out = np.sum(conv_features, axis=2)

    np.testing.assert_allclose(cfconv_layer_out, cfconv_manual_out)


def test_interaction_block():
    # Tests the correct output shape of an interaction block
    interaction_b = InteractionBlock(test_nbh,
                                     num_inputs=num_feats,
                                     num_gaussians=num_gaussians,
                                     num_filters=num_filters)
    rbf_out = rbf_layer(distances)
    interaction_output = interaction_b(test_features, rbf_out)

    np.testing.assert_equal(interaction_output.shape,
                            (num_examples, num_beads, num_filters))


def test_schnet_block():
    # Tests proper forwarding through SchNet wrapper class
    interaction = InteractionBlock(test_nbh,
                                   num_inputs=num_feats,
                                   num_gaussians=num_gaussians, num_filters=num_filters)

    schnet_block = SchnetBlock(interaction, rbf_layer)
    output = schnet_block(test_features, distances)
    assert output.size() == test_features.size()

