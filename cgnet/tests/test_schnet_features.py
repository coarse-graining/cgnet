# Author: Dominik Lemm

import numpy as np
import torch

from cgnet.feature import (ContinuousFilterConvolution, InteractionBlock,
                           SchnetFeature, CGBeadEmbedding)

# Define sizes for a pseudo-dataset
frames = np.random.randint(10, 30)
beads = np.random.randint(5, 10)

# Define random feature sizes
n_gaussians = np.random.randint(5, 10)
n_feats = np.random.randint(4, 16)
n_filters = np.random.randint(4, 16)

# Create random test input features, radial basis function output and
# continuous convolution feature
test_rbf = torch.randn((frames, beads, beads - 1, n_gaussians))
test_features = torch.randn((frames, beads, n_feats))
test_cfconv_features = torch.randn((frames, beads, n_filters))

# Create a simple neighbor list in which all beads see each other
# Shape (num_frames, num_beads, num_beads -1)
test_nbh = np.tile(np.arange(beads), (frames, beads, 1))
inverse_identity = np.eye(beads, beads) != 1
test_nbh = torch.from_numpy(test_nbh[:, inverse_identity].reshape(frames,
                                                                  beads,
                                                                  beads - 1))


def test_continuous_convolution():
    # Comparison of the ContinuousFilterConvolution with a manual numpy calculation

    # Calculate continuous convolution output with the created layer
    cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                         n_filters=n_filters)

    # Calculate convolution manually
    n_neighbors = beads - 1
    test_nbh_np = test_nbh.numpy()
    test_feat_np = test_cfconv_features.numpy()

    # Feature tensor needs to be transformed from
    # (n_frames, n_beads, n_features)
    # to
    # (n_frames, n_beads, n_neighbors, n_features)
    # This can be done by feeding the features of a respective bead into
    # its position in the neighbor_list.
    # Gather the features into the respective places in the neighbor list
    neighbor_list = test_nbh_np.reshape(-1, beads * n_neighbors, 1)
    neighbor_list = neighbor_list.repeat(n_filters, axis=2)
    # Gather the features into the respective places in the neighbor list
    neighbor_features = np.take_along_axis(test_feat_np, neighbor_list, axis=1)
    # Reshape back to (n_frames, n_beads, n_neighbors, n_features) for
    # element-wise multiplication with the filter
    neighbor_features = neighbor_features.reshape(frames, beads,
                                                  n_neighbors, -1)

    # In order to compare the layer output with the manual calculation, we
    # need to use the same filter generator (2 linear layers with and without
    # activation function, respectively).
    test_conv_filter = cfconv.filter_generator(test_rbf).detach().numpy()

    # element-wise multiplication and pooling
    conv_features = neighbor_features * test_conv_filter
    cfconv_manual_out = np.sum(conv_features, axis=2)

    np.testing.assert_allclose(cfconv_layer_out, cfconv_manual_out)


def test_interaction_block():
    # Tests the correct output shape of an interaction block
    interaction_b = InteractionBlock(n_inputs=n_feats,
                                     n_gaussians=n_gaussians,
                                     n_filters=n_filters)
    interaction_output = interaction_b(test_features, test_rbf, test_nbh)

    np.testing.assert_equal(interaction_output.shape,
                            (frames, beads, n_filters))


def test_shared_weights():
    # Tests the weight sharing functionality of the interaction block
    feature_size = np.random.randint(4, 8)
    embedding_layer = CGBeadEmbedding(n_embeddings=2,
                                      embedding_dim=2)
    # Initialize two Schnet networks
    # With and without weight sharing, respectively.
    schnet_feature_no_shared_weights = SchnetFeature(feature_size=feature_size,
                                                     embedding_layer=embedding_layer,
                                                     n_interaction_blocks=2,
                                                     share_weights=False)
    schnet_feature_shared_weights = SchnetFeature(feature_size=feature_size,
                                                  embedding_layer=embedding_layer,
                                                  n_interaction_blocks=2,
                                                  share_weights=True)

    # Loop over all parameters in both interaction blocks
    # If the weights are shared, the parameters in both interaction blocks
    # should be equal, respectively.
    for param1, param2 in zip(
            schnet_feature_shared_weights.interaction_blocks[0].parameters(),
            schnet_feature_shared_weights.interaction_blocks[1].parameters()):
        assert np.array_equal(param1.detach().numpy(), param2.detach().numpy())

    # If the weights are not shared, the parameters should be different.
    for param1, param2 in zip(
            schnet_feature_no_shared_weights.interaction_blocks[0].parameters(),
            schnet_feature_no_shared_weights.interaction_blocks[1].parameters()):
        assert not np.array_equal(param1.detach().numpy(),
                                  param2.detach().numpy())


def test_schnet_feature():
    # TODO: Will be implemented once the SchnetFeature is fully functional
    # Tests proper forwarding through SchNet wrapper class
    # interaction = InteractionBlock(num_inputs=num_feats,
    #                                num_gaussians=num_gaussians,
    #                                num_filters=num_filters)
    #
    # schnet_block = SchnetFeature(interaction, rbf_layer)
    # output = schnet_block(test_features, distances)
    # assert output.size() == test_features.size()
    #
    pass