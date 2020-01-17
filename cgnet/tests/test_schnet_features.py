# Author: Dominik Lemm
# Contributors: Nick Charron, Brooke Husic

import numpy as np
import torch

from cgnet.feature import (ContinuousFilterConvolution, InteractionBlock,
                           SchnetFeature, CGBeadEmbedding, GeometryStatistics,
                           Geometry)

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
    assert schnet_feature._distance_pairs == geom_stats._distance_pairs
    np.testing.assert_equal(schnet_feature.redundant_distance_mapping,
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


def test_schnet_with_dummy_atoms_is_different():
    # Tests whether forwarding through a schnet feature returns a different
    # result when some atoms are dummy atoms
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

    # Force the embedding property to have zeros
    embedding_property_dummy = (embedding_property -
                                torch.min(embedding_property))
    schnet_features_dummy = schnet_feature(torch.from_numpy(coords),
                                           embedding_property_dummy)

    # These arrays shouldn't be equal because one has dummy atoms and
    # one doesn't
    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_equal,
                             schnet_features.detach().numpy(),
                             schnet_features_dummy.detach().numpy())


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
