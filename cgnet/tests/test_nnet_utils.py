# Authors: Nick Charron, Brooke Husic

import copy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from cgnet.network import (lipschitz_projection, dataset_loss, CGnet,
                           ForceLoss, Simulation)
from cgnet.network.utils import _schnet_feature_linear_extractor
from cgnet.feature import (MoleculeDataset, LinearLayer, SchnetFeature,
                           CGBeadEmbedding, GeometryFeature, FeatureCombiner,
                           GeometryStatistics, RadialBasisFunction)
from nose.tools import assert_raises

# Here we create testing data from a random linear protein
# with a random number of frames
frames = np.random.randint(5, 10)
beads = np.random.randint(4, 10)
dims = 3

coords = np.random.randn(frames, beads, dims).astype('float32')
forces = np.random.randn(frames, beads, dims).astype('float32')
num_embeddings = np.random.randint(2, 10)
embedding_array = np.random.randint(1, num_embeddings, size=beads)
# beadwise embeddings
embeddings = np.tile(embedding_array, [coords.shape[0], 1])

# Here we instance statistics in order ot construct a feature combiner
# in the test_lipschitz_full_model_mask test. We need the distance indices
stats = GeometryStatistics(coords, backbone_inds='all', get_all_distances=True)
dist_idx = stats.return_indices('Distances')

# Here, we instance a molecular dataset with sampler and dataloader
sampler = SubsetRandomSampler(np.arange(0, frames, 1))
dataset = MoleculeDataset(coords, forces)
batch_size = np.random.randint(2, high=4)
loader = DataLoader(dataset, sampler=sampler,
                    batch_size=batch_size)
schnet_dataset = MoleculeDataset(coords, forces, embeddings)
schnet_loader = DataLoader(schnet_dataset, sampler=sampler,
                           batch_size=np.random.randint(2, high=10))

# Here we construct a single hidden layer architecture with random
# widths and a terminal contraction to a scalar output
arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))

# Here we construct a CGnet model using the above architecture
# as well as variables to be used in CG simulation tests
model = CGnet(arch, ForceLoss()).float()
model.eval()

# Here, we set of the parameters of the SchnetFeature
# for dataset_loss tests below

feature_size = np.random.randint(5, 10)  # random feature size
embedding_dim = beads  # embedding property size
n_interaction_blocks = np.random.randint(1, 3)  # random number of interactions
neighbor_cutoff = np.random.uniform(0, 1)  # random neighbor cutoff
# random embedding property
embedding_layer = CGBeadEmbedding(n_embeddings=num_embeddings,
                                  embedding_dim=feature_size)
rbf_layer = RadialBasisFunction()

# Here we use the above variables to create the SchnetFeature
schnet_feature = SchnetFeature(feature_size=feature_size,
                               embedding_layer=embedding_layer,
                               rbf_layer=rbf_layer,
                               n_interaction_blocks=n_interaction_blocks,
                               calculate_geometry=True,
                               n_beads=beads,
                               neighbor_cutoff=neighbor_cutoff)

# architecture to match schnet_feature output
schnet_arch = (LinearLayer(feature_size, dims, activation=nn.Tanh()) +
               LinearLayer(dims, 1, activation=None))
schnet_model = CGnet(schnet_arch, ForceLoss(), feature=schnet_feature)

lipschitz_strength = 1


def _regularization_function(model, strength=lipschitz_strength):
    lipschitz_projection(model, strength=strength)


def test_lipschitz_weak_and_strong():
    # Test proper functioning of strong lipschitz projection ( lambda_ << 1 )
    # Strongly projected weights should have greatly reduced magnitudes

    # Here we create a single layer test architecture and use it to
    # construct a simple CGnet model. We use a random hidden layer width
    width = np.random.randint(10, high=20)
    test_arch = (LinearLayer(1, width, activation=nn.Tanh()) +
                 LinearLayer(width, 1, activation=None))
    test_model = CGnet(test_arch, ForceLoss()).float()

    # Here we set the lipschitz projection to be extremely strong ( lambda_ << 1 )
    lambda_ = float(1e-12)

    # We save the numerical values of the pre-projection weights, perform the
    # strong lipschitz projection, and verify that the post-projection weights
    # are greatly reduced in magnitude.
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    lipschitz_projection(test_model, lambda_)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                               if isinstance(layer, nn.Linear)]
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal, pre, post)
        assert np.linalg.norm(pre) > np.linalg.norm(post)

    # Next, we test weak lipschitz projection ( lambda_ >> 1 )
    # A weak Lipschitz projection should leave weights entirely unchanged
    # This test is identical to the one above, except here we verify that
    # the post-projection weights are unchanged by the lipshchitz projection
    lambda_ = float(1e12)
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    lipschitz_projection(test_model, lambda_)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                               if isinstance(layer, nn.Linear)]
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_array_equal(pre.numpy(), post.numpy())


def test_lipschitz_cgnet_network_mask():
    # Test lipschitz mask functionality for random binary vanilla cgnet network
    # mask using strong Lipschitz projection ( lambda_ << 1 )
    # If the mask element is True, a strong Lipschitz projection
    # should occur - else, the weights should remain unchanged.

    # Here we create a 10 layer hidden architecture with a
    # random width, and create a subsequent CGnet model. For
    width = np.random.randint(10, high=20)
    test_arch = LinearLayer(1, width, activation=nn.Tanh())
    for _ in range(9):
        test_arch += LinearLayer(width, width, activation=nn.Tanh())
    test_arch += LinearLayer(width, 1, activation=None)
    test_model = CGnet(test_arch, ForceLoss()).float()
    lambda_ = float(1e-12)
    pre_projection_cgnet_weights = [layer.weight.data for layer in test_model.arch
                                    if isinstance(layer, nn.Linear)]

    # Next, we create a random binary lipschitz mask, which prevents lipschitz
    # projection for certain random layers
    lip_mask = [np.random.randint(2, dtype=bool) for _ in test_arch
                if isinstance(_, nn.Linear)]
    lipschitz_projection(test_model, lambda_, network_mask=lip_mask)
    post_projection_cgnet_weights = [layer.weight.data for layer in test_model.arch
                                     if isinstance(layer, nn.Linear)]
    # Here we verify that the masked layers remain unaffected by the strong
    # Lipschitz projection
    for mask_element, pre, post in zip(lip_mask, pre_projection_cgnet_weights,
                                       post_projection_cgnet_weights):
        # If the mask element is True then the norm of the weights should be greatly
        # reduced after the lipschitz projection
        if mask_element:
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     pre.numpy(), post.numpy())
            assert np.linalg.norm(pre.numpy()) > np.linalg.norm(post.numpy())
        # If the mask element is False then the weights should be unaffected
        if not mask_element:
            np.testing.assert_array_equal(pre.numpy(), post.numpy())


def test_schnet_weight_extractor():
    # Tests the hidden helper method, _schnet_feature_linear_extractor()
    # There should be 5 nn.Linear instances per interaction block in a
    # SchnetFeature. We use the random SchnetFeature created in the top
    # of this file

    # First we test to see if the layers extracted are in fact instances
    # of nn.Linear
    linear_list = _schnet_feature_linear_extractor(schnet_feature)
    for layer in linear_list:
        assert isinstance(layer, nn.Linear)
    # Next, we assert that the number of nn.Linear instances are
    # equal to 5 x n_interaction_blocks, because each interaction
    # block has 5 nn.Linear instances
    assert len(linear_list) == 5 * len(schnet_feature.interaction_blocks)


def test_lipschitz_schnet_mask():
    # Test lipschitz mask functionality for random binary schnet mask
    # Using strong Lipschitz projection ( lambda_ << 1 )
    # If the mask element is True, a strong Lipschitz projection
    # should occur - else, the weights should remain unchanged.

    # Here we ceate a CGSchNet model with 10 interaction blocks
    # and a random feature size, embedding, and cutoff from the
    # setup at the top of this file with no terminal network
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   rbf_layer=rbf_layer,
                                   n_interaction_blocks=10,
                                   calculate_geometry=True,
                                   n_beads=beads,
                                   neighbor_cutoff=neighbor_cutoff)

    # single weight layer at the end to contract down to an energy
    schnet_test_arch = [LinearLayer(feature_size, 1, activation=None)]
    schnet_test_model = CGnet(schnet_arch, ForceLoss(), feature=schnet_feature)

    lambda_ = float(1e-12)
    pre_projection_schnet_weights = _schnet_feature_linear_extractor(schnet_test_model.feature,
                                                                     return_weight_data_only=True)
    # Convert torch tensors to numpy arrays for testing
    pre_projection_schnet_weights = [weight
                                     for weight in pre_projection_schnet_weights]
    # Next, we create a random binary lipschitz mask, which prevents lipschitz
    # projection for certain random schnet layers. There are 5 instances of
    # nn.Linear for each schnet interaction block
    lip_mask = [np.random.randint(2, dtype=bool)
                for _ in range(5 * len(schnet_feature.interaction_blocks))]

    # Here we make the lipschitz projection
    lipschitz_projection(schnet_test_model, lambda_, schnet_mask=lip_mask)
    post_projection_schnet_weights = _schnet_feature_linear_extractor(schnet_test_model.feature,
                                                                      return_weight_data_only=True)
    # Convert torch tensors to numpy arrays for testing
    post_projection_schnet_weights = [weight
                                      for weight in post_projection_schnet_weights]
    # Here we verify that the masked layers remain unaffected by the strong
    # Lipschitz projection
    for mask_element, pre, post in zip(lip_mask, pre_projection_schnet_weights,
                                       post_projection_schnet_weights):
        # If the mask element is True then the norm of the weights should be greatly
        # reduced after the lipschitz projection
        if mask_element:
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     pre.numpy(), post.numpy())
            assert np.linalg.norm(pre.numpy()) > np.linalg.norm(post.numpy())
        # If the mask element is False then the weights should be unaffected
        if not mask_element:
            np.testing.assert_array_equal(pre.numpy(), post.numpy())


def test_lipschitz_full_model_random_mask():
    # Test lipschitz mask functionality for random binary schnet mask
    # and random binary terminal network mask for a model that contains
    # both SchnetFeatures and a terminal network
    # using strong Lipschitz projection ( lambda_ << 1 )
    # If the mask element is True, a strong Lipschitz projection
    # should occur - else, the weights should remain unchanged.

    # Here we ceate a CGSchNet model with a GeometryFeature layer,
    # 10 interaction blocks, a random feature size, embedding, and
    # cutoff from the setup at the top of this file, and a terminal
    # network of 10 layers and with a random width
    width = np.random.randint(10, high=20)
    test_arch = LinearLayer(feature_size, width, activation=nn.Tanh())
    for _ in range(9):
        test_arch += LinearLayer(width, width, activation=nn.Tanh())
    test_arch += LinearLayer(width, 1, activation=None)
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   rbf_layer=rbf_layer,
                                   n_interaction_blocks=10,
                                   n_beads=beads,
                                   neighbor_cutoff=neighbor_cutoff,
                                   calculate_geometry=False)
    feature_list = FeatureCombiner([GeometryFeature(feature_tuples='all_backbone',
                                                    n_beads=beads), schnet_feature],
                                   distance_indices=dist_idx)
    full_test_model = CGnet(test_arch, ForceLoss(), feature=feature_list)

    # The pre_projection weights are the terminal network weights followed by
    # the SchnetFeature weights
    lambda_ = float(1e-12)
    pre_projection_terminal_network_weights = [layer.weight.data
                                               for layer in full_test_model.arch
                                               if isinstance(layer, nn.Linear)]
    pre_projection_schnet_weights = _schnet_feature_linear_extractor(full_test_model.feature.layer_list[-1],
                                                                     return_weight_data_only=True)
    full_pre_projection_weights = (pre_projection_terminal_network_weights +
                                   pre_projection_schnet_weights)

    # Next, we assemble the masks for both the terminal network and the
    # SchnetFeature weights. There are 5 instances of nn.Linear for each
    # interaction block in the SchnetFeature
    network_lip_mask = [np.random.randint(2, dtype=bool)
                        for _ in range(len([layer for layer in full_test_model.arch
                                            if isinstance(layer, nn.Linear)]))]

    schnet_lip_mask = [np.random.randint(2, dtype=bool)
                       for _ in range(5 * len(schnet_feature.interaction_blocks))]
    full_lip_mask = network_lip_mask + schnet_lip_mask

    # Here we make the lipschitz projection
    lipschitz_projection(full_test_model, lambda_, network_mask=network_lip_mask,
                         schnet_mask=schnet_lip_mask)
    post_projection_terminal_network_weights = [layer.weight.data
                                                for layer in full_test_model.arch
                                                if isinstance(layer, nn.Linear)]
    post_projection_schnet_weights = _schnet_feature_linear_extractor(full_test_model.feature.layer_list[-1],
                                                                      return_weight_data_only=True)
    full_post_projection_weights = (post_projection_terminal_network_weights +
                                    post_projection_schnet_weights)

    # Here we verify that the masked layers remain unaffected by the strong
    # Lipschitz projection
    for mask_element, pre, post in zip(full_lip_mask, full_pre_projection_weights,
                                       full_post_projection_weights):
        # If the mask element is True then the norm of the weights should be greatly
        # reduced after the lipschitz projection
        if mask_element:
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     pre.numpy(), post.numpy())
            assert np.linalg.norm(pre.numpy()) > np.linalg.norm(post.numpy())
        # If the mask element is False then the weights should be unaffected
        if not mask_element:
            np.testing.assert_array_equal(pre.numpy(), post.numpy())


def test_lipschitz_full_model_all_mask():
    # Test lipschitz mask functionality for completely False schnet mask
    # and completely False terminal network mask for a model that contains
    # both SchnetFeatures and a terminal network
    # using strong Lipschitz projection ( lambda_ << 1 )
    # In this case, we expect all weight layers to remain unchanged

    # Here we ceate a CGSchNet model with a GeometryFeature layer,
    # 10 interaction blocks, a random feature size, embedding, and
    # cutoff from the setup at the top of this file, and a terminal
    # network of 10 layers and with a random width
    width = np.random.randint(10, high=20)
    test_arch = LinearLayer(feature_size, width, activation=nn.Tanh())
    for _ in range(9):
        test_arch += LinearLayer(width, width, activation=nn.Tanh())
    test_arch += LinearLayer(width, 1, activation=None)
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   rbf_layer=rbf_layer,
                                   n_interaction_blocks=10,
                                   n_beads=beads,
                                   neighbor_cutoff=neighbor_cutoff,
                                   calculate_geometry=False)
    feature_list = FeatureCombiner([GeometryFeature(feature_tuples='all_backbone',
                                                    n_beads=beads), schnet_feature],
                                   distance_indices=dist_idx)
    full_test_model = CGnet(test_arch, ForceLoss(), feature=feature_list)

    # The pre_projection weights are the terminal network weights followed by
    # the SchnetFeature weights
    lambda_ = float(1e-12)
    pre_projection_terminal_network_weights = [layer.weight.data
                                               for layer in full_test_model.arch
                                               if isinstance(layer, nn.Linear)]
    pre_projection_schnet_weights = _schnet_feature_linear_extractor(full_test_model.feature.layer_list[-1],
                                                                     return_weight_data_only=True)
    full_pre_projection_weights = (pre_projection_terminal_network_weights +
                                   pre_projection_schnet_weights)

    # Here we make the lipschitz projection, specifying the 'all' option for
    # both the terminal network mask and the schnet mask
    lipschitz_projection(full_test_model, lambda_, network_mask='all',
                         schnet_mask='all')
    post_projection_terminal_network_weights = [layer.weight.data
                                                for layer in full_test_model.arch
                                                if isinstance(layer, nn.Linear)]
    post_projection_schnet_weights = _schnet_feature_linear_extractor(full_test_model.feature.layer_list[-1],
                                                                      return_weight_data_only=True)
    full_post_projection_weights = (post_projection_terminal_network_weights +
                                    post_projection_schnet_weights)

    # Here we verify that all weight layers remain unaffected by the strong
    # Lipschitz projection
    for pre, post in zip(full_pre_projection_weights,
                         full_post_projection_weights):
        np.testing.assert_array_equal(pre.numpy(), post.numpy())


def test_dataset_loss():
    # Test dataset loss by comparing results from different batch sizes
    # The loss calculated over the entire dataset should not be affected
    # by the batch size used by the dataloader. This test uses a standard
    # CGnet model

    # First, we get dataset loss using the greater-than-one batch size
    # loader from the preamble
    loss = dataset_loss(model, loader, train_mode=False)

    # Next, we do the same but use a loader with a batch size of 1
    single_point_loader = DataLoader(dataset, sampler=sampler,
                                     batch_size=1)
    single_point_loss = dataset_loss(model, single_point_loader,
                                     train_mode=False)

    # Here, we verify that the two losses over the dataset are equal
    np.testing.assert_allclose(loss, single_point_loss, rtol=1e-5)


def test_dataset_loss_model_modes():
    # Test whether models are returned to train mode after eval is specified

    # Define mode to pass into the dataset
    model_dataset = CGnet(copy.deepcopy(arch), ForceLoss()).float()
    # model should be in training mode by default
    assert model_dataset.training == True

    # Simple datalaoder
    loader = DataLoader(dataset, batch_size=batch_size)

    loss_dataset = dataset_loss(model_dataset,
                                loader, train_mode=False)

    # The model should be returned to the default train state
    assert model_dataset.training == True


def test_dataset_loss_with_optimizer():
    # Test manual batch processing vs. dataset_loss during training
    # Make a simple model and test that a manual on-the-fly loss calculation
    # approximately matches the one from dataset_loss when given an optimizer

    # Set up the network
    num_epochs = 5

    # Empty lists to be compared after training
    epochal_train_losses_manual = []
    epochal_train_losses_dataset = []

    # We require two models and two optimizers to keep things separate
    # The architectures MUST be deep copied or else they are tethered
    # to each other
    model_manual = CGnet(copy.deepcopy(arch), ForceLoss()).float()
    model_dataset = CGnet(copy.deepcopy(arch), ForceLoss()).float()

    optimizer_manual = torch.optim.Adam(model_manual.parameters(),
                                        lr=1e-5)
    optimizer_dataset = torch.optim.Adam(model_dataset.parameters(),
                                         lr=1e-5)

    # We want a nonrandom loader so we can compare the losses at the end
    nonrandom_loader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(1, num_epochs+1):
        train_loss_manual = 0.0
        train_loss_dataset = 0.0

        # This is the manual part
        effective_batch_num = 0

        for batch_num, batch_data in enumerate(nonrandom_loader):
            optimizer_manual.zero_grad()
            coord, force, embedding_property = batch_data

            if batch_num == 0:
                ref_batch_size = coord.numel()

            batch_weight = coord.numel() / ref_batch_size

            energy, pred_force = model_manual.forward(coord,
                                                      embedding_property)

            batch_loss = model_manual.criterion(pred_force, force)
            batch_loss.backward()
            optimizer_manual.step()

            train_loss_manual += batch_loss.detach().cpu() * batch_weight
            effective_batch_num += batch_weight

        train_loss_manual = train_loss_manual / effective_batch_num
        epochal_train_losses_manual.append(train_loss_manual.numpy())

        # This is the dataset loss part
        train_loss_dataset = dataset_loss(model_dataset,
                                          nonrandom_loader,
                                          optimizer_dataset)
        epochal_train_losses_dataset.append(train_loss_dataset)

    np.testing.assert_allclose(epochal_train_losses_manual,
                               epochal_train_losses_dataset,
                               rtol=1e-4)


def test_dataset_loss_with_optimizer_and_regularization():
    # Test manual batch processing vs. dataset_loss during regularized training
    # Make a simple model and test that a manual on-the-fly loss calculation
    # approximately matches the one from dataset_loss when given an optimizer
    # and regularization function

    # Set up the network
    num_epochs = 5

    # Empty lists to be compared after training
    epochal_train_losses_manual = []
    epochal_train_losses_dataset = []

    # We require two models and two optimizers to keep things separate
    # The architectures MUST be deep copied or else they are tethered
    # to each other
    model_manual = CGnet(copy.deepcopy(arch), ForceLoss()).float()
    model_dataset = CGnet(copy.deepcopy(arch), ForceLoss()).float()

    optimizer_manual = torch.optim.Adam(model_manual.parameters(),
                                        lr=1e-5)
    optimizer_dataset = torch.optim.Adam(model_dataset.parameters(),
                                         lr=1e-5)

    # We want a nonrandom loader so we can compare the losses at the end
    nonrandom_loader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(1, num_epochs+1):
        train_loss_manual = 0.0
        train_loss_dataset = 0.0

        # This is the manual part
        effective_batch_num = 0

        for batch_num, batch_data in enumerate(nonrandom_loader):
            optimizer_manual.zero_grad()
            coord, force, embedding_property = batch_data

            if batch_num == 0:
                ref_batch_size = coord.numel()

            batch_weight = coord.numel() / ref_batch_size

            energy, pred_force = model_manual.forward(coord,
                                                      embedding_property)

            batch_loss = model_manual.criterion(pred_force, force)
            batch_loss.backward()
            optimizer_manual.step()

            lipschitz_projection(model_manual, strength=lipschitz_strength)

            train_loss_manual += batch_loss.detach().cpu() * batch_weight
            effective_batch_num += batch_weight

        train_loss_manual = train_loss_manual / effective_batch_num
        epochal_train_losses_manual.append(train_loss_manual.numpy())

        # This is the dataset loss part
        train_loss_dataset = dataset_loss(model_dataset,
                                          nonrandom_loader,
                                          optimizer_dataset,
                                          _regularization_function)
        epochal_train_losses_dataset.append(train_loss_dataset)

    np.testing.assert_allclose(epochal_train_losses_manual,
                               epochal_train_losses_dataset,
                               rtol=1e-4)


def test_schnet_dataset_loss():
    # Test dataset loss by comparing results from different batch sizes
    # The loss calculated over the entire dataset should not be affected
    # by the batch size used by the dataloader. This test uses a CGnet
    # with a SchnetFeature

    # First, we get dataset loss using the greater-than-one batch size
    # loader from the preamble
    loss = dataset_loss(schnet_model, schnet_loader, train_mode=False)

    # Next, we do the same but use a loader with a batch size of 1
    single_point_loader = DataLoader(schnet_dataset, sampler=sampler,
                                     batch_size=1)
    single_point_loss = dataset_loss(schnet_model, single_point_loader,
                                     train_mode=False)

    # Here, we verify that the two losses over the dataset are equal
    np.testing.assert_allclose(loss, single_point_loss, rtol=1e-5)
