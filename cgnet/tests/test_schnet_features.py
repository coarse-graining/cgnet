# Author: Dominik Lemm

import numpy as np
import torch

from cgnet.feature import ContinuousFilterConvolution, InteractionBlock

# Random protein
num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)

# Random feature sizes
num_gaussians = np.random.randint(5, 10)
num_feats = np.random.randint(4, 16)
num_filters = np.random.randint(4, 16)

# Create test input features, radial basis function output and neighbor list
test_rbf = torch.randn((num_examples, num_beads, num_beads - 1, num_gaussians))
test_features = torch.randn((num_examples, num_beads, num_feats))
test_cfconv_features = torch.randn((num_examples, num_beads, num_filters))

# Create a simple neighbor list in which all beads see each other
# Shape (num_examples, num_beads, num_beads -1)
test_nbh = np.tile(np.arange(num_beads), (num_examples, num_beads, 1))
inverse_identity = np.eye(num_beads, num_beads) != 1
test_nbh = torch.from_numpy(test_nbh[:, inverse_identity].reshape(num_examples,
                                                                  num_beads,
                                                                  num_beads - 1))


def test_continuous_convolution():
    # Calculate continuous convolution output with the created layer
    cfconv = ContinuousFilterConvolution(num_gaussians=num_gaussians,
                                         num_filters=num_filters)
    cfconv_layer_out = cfconv.forward(test_cfconv_features, test_rbf,
                                      test_nbh).detach()

    # Calculate convolution manually
    test_conv_filter = cfconv.filter_generator(test_rbf).detach().numpy()

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
    interaction_b = InteractionBlock(num_inputs=num_feats,
                                     num_gaussians=num_gaussians,
                                     num_filters=num_filters)
    interaction_output = interaction_b(test_features, test_rbf, test_nbh)

    np.testing.assert_equal(interaction_output.shape,
                            (num_examples, num_beads, num_filters))
