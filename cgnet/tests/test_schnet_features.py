# Author: Dominik Lemm
# Contributors: Nick Charron

from nose.tools import assert_raises
import numpy as np
import torch
import torch.nn as nn

from cgnet.feature import (ContinuousFilterConvolution, InteractionBlock,
                           SchnetFeature, CGBeadEmbedding, GeometryStatistics,
                           Geometry, ShiftedSoftplus, SimpleNormLayer,
                           NeighborNormLayer)

g = Geometry(method='torch')

# Define sizes for a pseudo-dataset
frames = np.random.randint(10, 30)
beads = np.random.randint(5, 10)

# create random linear protein data
coords = np.random.randn(frames, beads, 3).astype(np.float32)

# Define random feature sizes
n_gaussians = np.random.randint(5, 10)
n_feats = np.random.randint(4, 16)
n_filters = np.random.randint(4, 16)
n_embeddings = np.random.randint(3, 5)
neighbor_cutoff = np.random.uniform(0, 1)

# Create random test input features, radial basis function output and
# continuous convolution feature
test_rbf = torch.randn((frames, beads, beads - 1, n_gaussians))


# Calculate redundant distances and create a simple neighbor list in which all
# beads see each other (shape [n_frames, n_beads, n_beads -1]).
_distance_pairs, _ = g.get_distance_indices(beads, [], [])
redundant_distance_mapping = g.get_redundant_distance_mapping(_distance_pairs)
distances = g.get_distances(
    _distance_pairs, torch.from_numpy(coords), norm=True)
distances = distances[:, redundant_distance_mapping]
test_nbh, test_nbh_mask = g.get_neighbors(distances, cutoff=neighbor_cutoff)


def test_continuous_convolution():
    # Comparison of the ContinuousFilterConvolution with a manual numpy calculation

    test_cfconv_features = torch.randn((frames, beads, n_filters))
    # Calculate continuous convolution output with the created layer
    cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                         n_filters=n_filters)
    cfconv_layer_out = cfconv.forward(test_cfconv_features, test_rbf,
                                      test_nbh, test_nbh_mask).detach()
    # Calculate convolution manually
    n_neighbors = beads - 1
    test_nbh_np = test_nbh.numpy()
    test_nbh_mask_np = test_nbh_mask.numpy()
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
    # Remove features from non-existing neighbors
    conv_features_masked = conv_features * test_nbh_mask_np[:, :, :, None]
    cfconv_manual_out = np.sum(conv_features_masked, axis=2)

    # Test if all the removed features are indeed 0
    assert not np.all(
        conv_features_masked[~test_nbh_mask_np.astype(np.bool)].astype(
            np.bool))
    # Test if the torch and numpy calculation are the same
    np.testing.assert_allclose(cfconv_layer_out, cfconv_manual_out)



def test_cfconv_bead_masking():
    # tests to see if the output of ContinousFilterConvolution
    # is properly masked if tha input has padding to account
    # for variable molecule sizes

    # First, we assmeble a tensor of mock padded embeddings
    # to emulate the effect of using a padded dataset
    variable_beads = np.random.randint(3, beads, size=frames) # random protein sizes
    variable_embeddings = [np.random.randint(1,
                           high=beads, size=bead) for bead in variable_beads]
    padded_embedding_list = []
    for embedding in variable_embeddings:
        pads_needed = beads - embedding.shape[0]
        padded_embeddings = np.hstack((embedding, np.zeros(pads_needed)))
        padded_embedding_list.append(padded_embeddings)
    embedding_property = torch.tensor(padded_embedding_list).long()
    bead_mask = torch.clamp(embedding_property, min=0, max=1).float()

    # We repeate the same procedure here as in test_continuous_convolution
    # But in addition we apply the bead_mask to the forward operation
    # and test to see if the artifical beads introduced by padding are 
    # zeroed out properly
    test_cfconv_features = torch.randn((frames, beads, n_filters))
    # Calculate continuous convolution output with the created layer
    cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                         n_filters=n_filters)
    cfconv_layer_out = cfconv.forward(test_cfconv_features, test_rbf,
                                      test_nbh, test_nbh_mask,
                                      bead_mask=bead_mask).detach()

    # Calculate convolution manually
    n_neighbors = beads - 1
    test_nbh_np = test_nbh.numpy()
    test_nbh_mask_np = test_nbh_mask.numpy()
    test_feat_np = test_cfconv_features.numpy()
    neighbor_list = test_nbh_np.reshape(-1, beads * n_neighbors, 1)
    neighbor_list = neighbor_list.repeat(n_filters, axis=2)
    neighbor_features = np.take_along_axis(test_feat_np, neighbor_list, axis=1)
    neighbor_features = neighbor_features.reshape(frames, beads,
                                                  n_neighbors, -1)
    test_conv_filter = cfconv.filter_generator(test_rbf).detach().numpy()
    conv_features = neighbor_features * test_conv_filter
    conv_features_masked = conv_features * test_nbh_mask_np[:, :, :, None]
    cfconv_manual_out = torch.tensor(np.sum(conv_features_masked, axis=2))

    # Here we mask the contributions from padded portions of the input
    cfconv_manual_out = cfconv_manual_out * bead_mask[:,:,None]

    # Test if the torch and numpy calculation are the same, including
    # after the filtering done by the bead_mask
    np.testing.assert_allclose(cfconv_layer_out, cfconv_manual_out.numpy())


def test_interaction_block():
    # Tests the correct output shape of an interaction block
    test_features = torch.randn((frames, beads, n_feats))
    interaction_b = InteractionBlock(n_inputs=n_feats,
                                     n_gaussians=n_gaussians,
                                     n_filters=n_filters)
    interaction_output = interaction_b(test_features, test_rbf, test_nbh,
                                       test_nbh_mask)

    np.testing.assert_equal(interaction_output.shape,
                            (frames, beads, n_filters))


def test_shared_weights():
    # Tests the weight sharing functionality of the interaction block
    feature_size = np.random.randint(4, 8)

    # Initialize two Schnet networks
    # With and without weight sharing, respectively.
    schnet_feature_no_shared_weights = SchnetFeature(feature_size=feature_size,
                                                     embedding_layer=None,
                                                     n_interaction_blocks=2,
                                                     n_beads=beads,
                                                     share_weights=False)
    schnet_feature_shared_weights = SchnetFeature(feature_size=feature_size,
                                                  embedding_layer=None,
                                                  n_interaction_blocks=2,
                                                  n_beads=beads,
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
            schnet_feature_no_shared_weights.interaction_blocks[0].parameters(
            ),
            schnet_feature_no_shared_weights.interaction_blocks[1].parameters()):
        assert not np.array_equal(param1.detach().numpy(),
                                  param2.detach().numpy())


def test_schnet_feature_geometry():
    # Tests SchnetFeature's calls to the Geometry class for
    # distance calculations
    # First, we instance a SchnetFeature that can call to Geometry
    schnet_feature = SchnetFeature(feature_size=n_feats,
                                   embedding_layer=None,
                                   n_interaction_blocks=2,
                                   calculate_geometry=True,
                                   n_beads=beads)

    # Next we instance a geom_stats that only calculates distances
    # and compare distance pair tuples
    geom_stats = GeometryStatistics(coords, backbone_inds='all',
                                    get_all_distances=True,
                                    get_backbone_angles=False,
                                    get_backbone_dihedrals=False,
                                    get_redundant_distance_mapping=True)

    # Here, we make sure that schnet_feature's calls to Geometry
    # can replicate those of a GeometryStatistics instance
    schnet_distance_pairs, _ = schnet_feature.geometry.get_distance_indices(beads,
                                                                         [], [])
    schnet_red_dist_map = schnet_feature.geometry.get_redundant_distance_mapping(
                                                            schnet_distance_pairs)
    assert schnet_distance_pairs == geom_stats._distance_pairs
    np.testing.assert_equal(schnet_red_dist_map,
                            geom_stats.redundant_distance_mapping)


def test_schnet_feature():
    # Tests proper forwarding through SchNet wrapper class

    # Create random embedding properties
    embedding_property = torch.randint(low=1, high=n_embeddings,
                                       size=(frames, beads))

    # Initialize the embedding and SchnetFeature class
    embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings,
                                      embedding_dim=n_feats)
    schnet_feature = SchnetFeature(feature_size=n_feats,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=2,
                                   calculate_geometry=True,
                                   n_beads=beads,
                                   neighbor_cutoff=neighbor_cutoff)

    # Next, we check that forwarding cartesian coordinates through SchnetFeature
    # produces the correct output feature sizes
    schnet_features = schnet_feature(torch.from_numpy(coords),
                                     embedding_property)
    assert schnet_features.size() == (frames, beads, n_feats)

    # To test the internal logic of SchnetFeature, a full forward pass is
    # computed using the same internal components as SchnetFeature.
    # First, the embedding feature and radial basis function expansion are
    # computed.
    features = embedding_layer.forward(embedding_property)
    rbf_expansion = schnet_feature.rbf_layer(distances=distances)

    # Then, the embedding features are used as an input to the first
    # interaction block and as a residual connection, respectively. Depending
    # on the amount of interaction blocks, the resulting features are then used
    # again as an input for the next interaction block and as a residual
    # connection, respectively.
    for interaction_block in schnet_feature.interaction_blocks:
        interaction_features = interaction_block(features=features,
                                                 rbf_expansion=rbf_expansion,
                                                 neighbor_list=test_nbh,
                                                 neighbor_mask=test_nbh_mask)
        features = features + interaction_features

    # The manually computed features and the feature from the forward pass
    # should be the same.
    np.testing.assert_allclose(schnet_features.detach(), features.detach())


def test_cg_embedding():
    # Tests if the embedding layer produces zero embeddings, same embeddings
    # for the same properties and different embeddings for different properties

    embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings,
                                      embedding_dim=n_feats)

    # Create a tensor full of zeroes
    zero_properties = torch.zeros(size=(frames, beads), dtype=torch.long)
    # Create a tensor with the same values
    same_properties = zero_properties + n_embeddings - 1
    # Create a tensor with random values
    random_properties = torch.randint_like(zero_properties, high=n_embeddings)

    # Test if passing zeroes produces an embedding full of zeroes
    zero_embedding = embedding_layer.forward(zero_properties).detach().numpy()
    np.testing.assert_equal(zero_embedding, 0.)

    # Test if passing the same value produces the same embedding
    same_embedding = embedding_layer.forward(same_properties).detach().numpy()
    assert np.all(same_embedding)

    # Test if passing different values produce different embeddings
    random_embedding = embedding_layer.forward(
        random_properties).detach().numpy()
    assert not np.all(random_embedding)


def test_schnet_activations():
    # Tests whether setting the activation function from the SchnetFeature
    # level correctly sets the activation for the InteractionBlocks and
    # ContinuousFilterConvolutions

    # Here, we instance some common activation functions and shuffle them
    alt_activations = [nn.Tanh(), nn.ReLU(), nn.ELU(), nn.Sigmoid()]
    alt_activation_classes = [nn.Tanh, nn.ReLU, nn.ELU, nn.Sigmoid]
    alt_lists = list(zip(alt_activations, alt_activation_classes))
    np.random.shuffle(alt_lists)
    alt_activations, alt_activation_classes = zip(*alt_lists)

    # Here we instance an random number of interaction blocks
    interaction_blocks = np.random.randint(1, high=5,
                                           size=len(alt_activations))

    # Here, we loop through all the activations and make sure that
    # they appear where they should in the model
    for activation, activation_class, iblock in zip(alt_activations,
                                                    alt_activation_classes,
                                                    interaction_blocks):
        schnet_feature = SchnetFeature(feature_size=n_feats,
                                       embedding_layer=None,
                                       activation=activation,
                                       n_interaction_blocks=iblock,
                                       calculate_geometry=True,
                                       n_beads=beads)
        # check all atom-wise layers and the filter generator networks
        # in both cases, the second index of the nn.Sequential objects
        # that hold the LinearLayers
        for interaction_block in schnet_feature.interaction_blocks:
            assert isinstance(interaction_block.output_dense[1],
                              activation_class)
            assert isinstance(interaction_block.cfconv.filter_generator[1],
                              activation_class)


def test_schnet_activation_default():
    # We check to see if the default activation, ShiftedSoftplus,
    # is correctly placed in the SchnetFeature

    interaction_blocks = np.random.randint(1, high=5)
    schnet_feature = SchnetFeature(feature_size=n_feats,
                                   embedding_layer=None,
                                   n_interaction_blocks=interaction_blocks,
                                   calculate_geometry=True,
                                   n_beads=beads)
    # check all atom-wise layers and the filter generator networks
    # in both cases, the second index of the nn.Sequential objects
    # that hold the LinearLayers

    for interaction_block in schnet_feature.interaction_blocks:
        assert isinstance(interaction_block.output_dense[1], ShiftedSoftplus)
        assert isinstance(interaction_block.cfconv.filter_generator[1],
                          ShiftedSoftplus)


def test_cfconv_simple_norm():
    # Tests to make sure simple bead number normalization on the output
    # of continuous filter convolution produces the expected numerical result

    test_cfconv_features = torch.randn((frames, beads, n_filters))
    # Create a random number for a simple normalization
    random_normalization = np.random.uniform(low=2.0,high=10.0)
    simple_norm = SimpleNormLayer(random_normalization)

    # Calculate continuous convolution output with the created layer
    cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                         normalization_layer=simple_norm,
                                         n_filters=n_filters)
    # Check to see if batchnorm is embedded properly in the cfconv
    assert isinstance(cfconv.normalization_layer, SimpleNormLayer)

    cfconv_layer_out = cfconv.forward(test_cfconv_features, test_rbf,
                                      test_nbh, test_nbh_mask).detach()
    # Calculate convolution manually
    n_neighbors = beads - 1
    test_nbh_np = test_nbh.numpy()
    test_nbh_mask_np = test_nbh_mask.numpy()
    test_feat_np = test_cfconv_features.numpy()

    # Feature tensor needs to be transformed from
    # (n_frames, n_beads, n_features) to
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
    # Remove features from non-existing neighbors
    conv_features_masked = conv_features * test_nbh_mask_np[:, :, :, None]
    cfconv_manual_out = np.sum(conv_features_masked, axis=2)

    # Test if all the removed features are indeed 0
    assert not np.all(
        conv_features_masked[~test_nbh_mask_np.astype(np.bool)].astype(
            np.bool))
    # Test if the torch and numpy calculation are the same
    normed_manual_out = torch.tensor(cfconv_manual_out) / random_normalization
    np.testing.assert_allclose(
        cfconv_layer_out, normed_manual_out.detach().numpy())


def test_cfconv_neighbor_norm():
    # Tests manual calculation of neighbor-normalized cfconv output with
    # the output of cfconv instanced with neighbor_norm=True.
    # This is the same test as above, but the output is normalized
    # by the number of neighbors

    test_cfconv_features = torch.randn((frames, beads, n_filters))
    # Calculate continuous convolution output with the created layer
    # using a NeighborNormLayer for normalization
    neighbor_norm_layer = NeighborNormLayer()
    cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                         normalization_layer=neighbor_norm_layer,
                                         n_filters=n_filters)
    assert isinstance(cfconv.normalization_layer, NeighborNormLayer)

    cfconv_layer_out = cfconv.forward(test_cfconv_features, test_rbf,
                                      test_nbh, test_nbh_mask).detach()
    # Calculate convolution manually
    n_neighbors = beads - 1
    test_nbh_np = test_nbh.numpy()
    test_nbh_mask_np = test_nbh_mask.numpy()
    test_feat_np = test_cfconv_features.numpy()

    neighbor_list = test_nbh_np.reshape(-1, beads * n_neighbors, 1)
    neighbor_list = neighbor_list.repeat(n_filters, axis=2)
    neighbor_features = np.take_along_axis(test_feat_np, neighbor_list, axis=1)
    neighbor_features = neighbor_features.reshape(frames, beads,
                                                  n_neighbors, -1)

    # In order to compare the layer output with the manual calculation, we
    # need to use the same filter generator (2 linear layers with and without
    # activation function, respectively).
    test_conv_filter = cfconv.filter_generator(test_rbf).detach().numpy()

    # element-wise multiplication and pooling
    conv_features = neighbor_features * test_conv_filter
    # Remove features from non-existing neighbors
    conv_features_masked = conv_features * test_nbh_mask_np[:, :, :, None]
    cfconv_manual_out = np.sum(conv_features_masked, axis=2)

    # normalize by number of neighbors:
    cfconv_manual_out = cfconv_manual_out / n_neighbors

    # Test if all the removed features are indeed 0
    assert not np.all(
        conv_features_masked[~test_nbh_mask_np.astype(np.bool)].astype(
            np.bool))
    # Test if the torch and numpy calculation are the same
    np.testing.assert_allclose(cfconv_layer_out, cfconv_manual_out)


def test_cfconv_batchnorm():
    # Tests the usage of batch normalization after application of the
    # continuous filter convolution

    test_cfconv_features = torch.randn((frames, beads, n_filters))
    # Calculate continuous convolution output with the created layer
    # Armed with a BatchNorm1d Layer
    batchnorm_layer = nn.BatchNorm1d(beads)
    cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                         n_filters=n_filters,
                                         normalization_layer=batchnorm_layer)
    # Check to see if batchnorm is embedded properly in the cfconv
    assert isinstance(cfconv.normalization_layer, nn.BatchNorm1d)

    cfconv_layer_out = cfconv.forward(test_cfconv_features, test_rbf,
                                      test_nbh, test_nbh_mask).detach()
    # Calculate convolution manually
    n_neighbors = beads - 1
    test_nbh_np = test_nbh.numpy()
    test_nbh_mask_np = test_nbh_mask.numpy()
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
    # Remove features from non-existing neighbors
    conv_features_masked = conv_features * test_nbh_mask_np[:, :, :, None]
    cfconv_manual_out = np.sum(conv_features_masked, axis=2)

    # Test if all the removed features are indeed 0
    assert not np.all(
        conv_features_masked[~test_nbh_mask_np.astype(np.bool)].astype(
            np.bool))
    # Test if the torch and numpy calculation are the same
    normed_manual_out = batchnorm_layer(torch.tensor(cfconv_manual_out))
    np.testing.assert_allclose(
        cfconv_layer_out, normed_manual_out.detach().numpy())
