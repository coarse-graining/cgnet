# Authors: Nick Charron, Brooke Husic

import numpy as np
import torch
import torch.nn as nn
from cgnet.feature import (FeatureCombiner, GeometryFeature, GeometryStatistics,
                           LinearLayer, SchnetFeature, CGBeadEmbedding)
from cgnet.network import (CGnet, ForceLoss, HarmonicLayer, ZscoreLayer,
                           Simulation)


def _get_random_schnet_feature(calculate_geometry=False):
    """ Helper function for producing SchnetFeature instances with
    random initializaitons

    Parameters
    ----------
    calculate_geometry : bool (default=False)
        specifies whether or not the returned SchnetFeature should
        calculate pairwise distances

    Returns
    -------
    SchnetFeature : SchnetFeature
        Instance of SchnetFeature with random initialization variables
    embedding_property: torch.Tensor
        Random embedding property for using in forwarding through
        FeatureCombiner or SchnetFeature in tests
    feature_size : int
        Random feature size for schnet interaction blocks
    """

    feature_size = np.random.randint(5, high=10)  # random feature size
    n_embeddings = np.random.randint(3, high=5)  # random embedding number
    embedding_dim = feature_size  # embedding property size
    n_interaction_blocks = np.random.randint(
        1, 3)  # random number of interactions
    neighbor_cutoff = np.random.uniform(0, 1)  # random neighbor cutoff
    # random embedding property
    embedding_property = torch.randint(low=1, high=n_embeddings,
                                       size=(n_frames, n_beads))
    embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings,
                                      embedding_dim=embedding_dim)
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=n_interaction_blocks,
                                   calculate_geometry=calculate_geometry,
                                   n_beads=n_beads,
                                   neighbor_cutoff=neighbor_cutoff)
    return schnet_feature, embedding_property, feature_size


def _get_random_architecture(input_size):
    """ Helper function to generate random hidden architecture for CGnet

    Parameters
    ----------
    input_size : int
        input dimension for the first LinearLayer of the hidden architecture

    Returns
    -------
    arch : list of LinearLayers
        Random hidden architecture for CGnet

    """

    depth = np.random.randint(3, high=6)  # random fully-connected depth
    width = np.random.randint(5, high=10)  # random fully-connected width
    arch = LinearLayer(input_size, width, activation=nn.Tanh())
    for i in range(2):
        arch += LinearLayer(width, width, activation=nn.Tanh())
    arch += LinearLayer(width, 1, activation=None)
    return arch


# First, we construct random linear protein data
n_beads = np.random.randint(5, high=10)
n_frames = np.random.randint(10, high=30)
coords_numpy = np.random.randn(n_frames, n_beads, 3)
coords_torch = torch.tensor(coords_numpy, requires_grad=True).float()

# Next, we recover the statistics of random protein data and we produce the
# means, constants, and callback indices of bonds for a HarmonicLayer prior
geom_stats = GeometryStatistics(coords_numpy, backbone_inds='all',
                                get_all_distances=True,
                                get_backbone_angles=False,
                                get_backbone_dihedrals=False,
                                get_redundant_distance_mapping=True)
bonds_list, _ = geom_stats.get_prior_statistics('Bonds', as_list=True)
bonds_idx = geom_stats.return_indices('Bonds')
# Here we use the bond statistics to create a HarmonicLayer
bond_potential = HarmonicLayer(bonds_idx, bonds_list)

# Next, we produce the zscore statistics and create a ZscoreLayer
zscores, _ = geom_stats.get_zscore_array()
zscore_layer = ZscoreLayer(zscores)

# Next, we create a GeometryFeature layer for the subsequent tests
# We only want it to calculate the distances, so we specify that
# the feature tuples are the ones we calculated in GeometryStatistics
geometry_feature = GeometryFeature(
    feature_tuples=geom_stats.master_description_tuples)


def test_combiner_geometry_feature():
    # Tests FeatureCombiner for just single GeometryFeature
    # In this case, geometry output should be None.
    # First, we instantiate a FeatureCombiner
    layer_list = [geometry_feature]
    feature_combiner = FeatureCombiner(layer_list,
                                       save_geometry=False)

    # If there is simply a GeometryFeature, then feature_combiner.forward()
    # should return feature_ouput, geometry_output, with geometry_features
    # equal to None
    feature_output, geometry_output = feature_combiner(coords_torch)
    assert feature_combiner.interfeature_transforms == [None]
    np.testing.assert_equal(list(feature_output.size()), list((n_frames,
                                                               len(geom_stats.master_description_tuples))))
    assert geometry_output is None


def test_combiner_schnet_feature():
    # Tests FeatureCombiner for just single SchnetFeature.
    # In this case, the geometry_output variable should be None.
    # We also test to make sure the energy/force output matches
    # that of a CGnet with the feature __init__ kwarg using the
    # SchnetFeature
    # First, we instantiate a FeatureCombiner with a SchnetFeature
    # That is capable of calculating pairwise distances (calculate_geometry
    # is True)
    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
        calculate_geometry=True)
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


def test_combiner_schnet_in_cgnet():
    # Here we test to see if a FeatureCombiner using just a SchnetFeature
    # produces the same output as a CGnet with a SchnetFeature for the
    # feature __init__ kwarg
    # First, we instantiate a FeatureCombiner with a SchnetFeature
    # That is capable of calculating pairwise distances (calculate_geometry
    # is True)
    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
        calculate_geometry=True)
    layer_list = [schnet_feature]
    feature_combiner = FeatureCombiner(layer_list)

    # Next, we make aa CGnet with a random hidden architecture
    arch = _get_random_architecture(feature_size)
    model = CGnet(arch, ForceLoss(), feature=feature_combiner)

    # Next, we forward the random protein data through the model
    # and assert the output has the correct shape
    energy, forces = model.forward(coords_torch,
                                   embedding_property=embedding_property)

    # Next, we make another CGnet with the same arch but embed a SchnetFeature
    # directly instead of using a FeatureCombiner
    model_2 = CGnet(arch, ForceLoss(), feature=schnet_feature)
    energy_2, forces_2 = model_2.forward(coords_torch,
                                         embedding_property=embedding_property)

    np.testing.assert_array_equal(energy.detach().numpy(),
                                  energy_2.detach().numpy())
    np.testing.assert_array_equal(forces.detach().numpy(),
                                  forces_2.detach().numpy())


def test_combiner_zscore():
    # Tests FeatureCombiner for proper forwarding with GeometryFeature and
    # ZscoreLayer, checking to make sure the output sizes are consistent
    # and that geometry_output is not None

    # First, we make our FeatureCombiner
    layer_list = [geometry_feature, zscore_layer]
    feature_combiner = FeatureCombiner(layer_list)

    # If there is simply a GeometryFeature, then feature_combiner.forward()
    # should return feature_ouput, geometry_output, with geometry_features
    # equal to None
    feature_output, geometry_output = feature_combiner(coords_torch)
    # Both transfroms should be None
    assert feature_combiner.interfeature_transforms == [None, None]
    np.testing.assert_equal(list(feature_output.size()),
                            list((n_frames,
                            len(geom_stats.master_description_tuples))))
    np.testing.assert_equal(list(geometry_output.size()),
                            list((n_frames,
                            len(geom_stats.master_description_tuples))))
    assert geometry_output is not None


def test_combiner_priors():
    # This test checks to see if the same energy/force results are obtained
    # using a FeatureCombiner instantiated with just a Geometry feature
    # as with a cgnet that uses a normal GeometryFeature as the feature
    # __init__ kwarg

    # First, we create our FeatureCombiner
    layer_list = [geometry_feature, zscore_layer]
    feature_combiner = FeatureCombiner(layer_list)

    # Next, we create CGnet and use the bond_potential prior and
    # feature_combiner.
    arch = _get_random_architecture(len(geom_stats.master_description_tuples))
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
    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
                                                          calculate_geometry=False)
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


def test_cgschnet_simulation_shapes():
    # Test simulation with embeddings and make sure the shapes of
    # the simulated coordinates, forces, and potential are correct
    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
                                                           calculate_geometry=True)
    layer_list = [schnet_feature]
    feature_combiner = FeatureCombiner(layer_list)

    # Next, we make aa CGnet with a random hidden architecture
    arch = _get_random_architecture(feature_size)
    model = CGnet(arch, ForceLoss(), feature=feature_combiner)

    sim_length = np.random.randint(10, 20)
    sim = Simulation(model, coords_torch, embedding_property, length=sim_length,
                     save_interval=1, beta=1., save_forces=True,
                     save_potential=True)

    traj = sim.simulate()

    np.testing.assert_array_equal(sim.simulated_traj.shape,
                                  [n_frames, sim_length, n_beads, 3])
    np.testing.assert_array_equal(sim.simulated_forces.shape,
                                  [n_frames, sim_length, n_beads, 3])
    np.testing.assert_array_equal(sim.simulated_potential.shape,
                                  [n_frames, sim_length, 1])


def test_feature_combiner_shapes():
    # Test feature combiner shapes with geometry features and schnet

    full_geometry_feature = GeometryFeature(feature_tuples='all_backbone',
                                            n_beads=n_beads)

    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
                                                          calculate_geometry=False)
    layer_list = [full_geometry_feature, schnet_feature]
    # grab distance indices
    dist_idx = geom_stats.return_indices('Distances')

    # Here, we set propagate_geometry to true
    feature_combiner = FeatureCombiner(layer_list, distance_indices=dist_idx,
                                       propagate_geometry=True)

    # The length of the geometry feature is the length of its tuples, where
    # each four-body dihedral is double counted to account for cosines and sines
    geom_feature_length = (len(full_geometry_feature.feature_tuples) +
                           len([f for f in full_geometry_feature.feature_tuples
                                if len(f) == 4]))

    # The total_size is what we need to input into our first linear layer, and
    # it represents the concatenation of the flatted schnet features with the
    # geometry features
    total_size = feature_size*n_beads + geom_feature_length

    # The forward method returns the object to be propagated to the NN and
    # the geometry features.
    feature_output, geometry_features = feature_combiner.forward(coords_torch,
                                                            embedding_property)

    np.testing.assert_array_equal(feature_output.shape,
                                  [n_frames, n_beads, feature_size])
    np.testing.assert_array_equal(geometry_features.shape,
                                  [n_frames, geom_feature_length])


def test_combiner_shape_with_geometry_propagation():
    # This tests a network with schnet features in which the geometry features
    # are also propagated through the neural network

    # This calculates all pairwise distances and backbone angles and dihedrals
    full_geometry_feature = GeometryFeature(feature_tuples='all_backbone',
                                            n_beads=n_beads)

    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
                                                          calculate_geometry=False)
    layer_list = [full_geometry_feature, schnet_feature]
    # grab distance indices
    dist_idx = geom_stats.return_indices('Distances')

    # Here, we set propagate_geometry to true
    feature_combiner = FeatureCombiner(layer_list, distance_indices=dist_idx,
                                       propagate_geometry=True)

    # The length of the geometry feature is the length of its tuples, where
    # each four-body dihedral is double counted to account for cosines and sines
    geom_feature_length = (len(full_geometry_feature.feature_tuples) +
                           len([f for f in full_geometry_feature.feature_tuples
                                if len(f) == 4]))

    # The total_size is what we need to input into our first linear layer, and
    # it represents the concatenation of the flatted schnet features with the
    # geometry features
    total_size = feature_size*n_beads + geom_feature_length

    # Now we just repeat the procedure from test_combiner_full above
    width = np.random.randint(5, high=10)  # random fully-connected width
    arch = LinearLayer(total_size,
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


def test_combiner_output_with_geometry_propagation():
    # This tests CGnet concatenation with propogating geometries
    # to make sure the FeatureCombiner method matches a manual calculation

    # This calculates all pairwise distances and backbone angles and dihedrals
    full_geometry_feature = GeometryFeature(feature_tuples='all_backbone',
                                            n_beads=n_beads)
    # Here we generate a random schent feature that does not calculate geometry
    schnet_feature, embedding_property, feature_size = _get_random_schnet_feature(
                                                          calculate_geometry=False)
    # grab distance indices
    dist_idx = geom_stats.return_indices('Distances')

    # Here we assemble the post-schnet fully connected network for manual
    # calculation of the energy/forces
    # The length of the geometry feature is the length of its tuples, where
    # each four-body dihedral is double counted to account for cosines and sines
    geom_feature_length = (len(full_geometry_feature.feature_tuples) +
                           len([f for f in full_geometry_feature.feature_tuples
                                if len(f) == 4]))
    total_size = feature_size*n_beads + geom_feature_length
    width = np.random.randint(5, high=10)  # random fully-connected width
    arch = LinearLayer(total_size,
                       width, activation=nn.Tanh())
    for i in range(2):
        arch += LinearLayer(width, width, activation=nn.Tanh())
    arch += LinearLayer(width, 1, activation=None)

    # Manual calculation using geometry feature concatenation and propagation
    # Here, we grab the distances to forward through the schnet feature. They
    # must be reindexed to the redundant mapping ammenable to schnet tools
    geometry_output = full_geometry_feature(coords_torch)
    distances = geometry_output[:, geom_stats.redundant_distance_mapping]
    schnet_output = schnet_feature(distances, embedding_property)

    # Here, we perform Manual feature concatenation between schnet and geometry
    # outputs. First, we flatten the schnet output for compatibility
    n_frames = coords_torch.shape[0]
    schnet_output = schnet_output.reshape(n_frames, -1)
    concatenated_features = torch.cat((schnet_output, geometry_output), dim=1)

    # Here, we feed the concatednated features through the terminal network and
    # predict the energy/forces
    terminal_network = nn.Sequential(*arch)
    manual_energy = terminal_network(concatenated_features)
    # Add in the bond potential contribution
    manual_energy += bond_potential(
        geometry_output[:, bond_potential.callback_indices])
    manual_forces = torch.autograd.grad(-torch.sum(manual_energy),
                                        coords_torch)[0]

    # Next, we produce the same output using a CGnet and test numerical
    # similarity, thereby testing the internal concatenation function of
    # CGnet.forward(). We create our model using a FeatureCombiner
    layer_list = [full_geometry_feature, schnet_feature]
    feature_combiner = FeatureCombiner(layer_list, distance_indices=dist_idx,
                                       propagate_geometry=True)

    model = CGnet(arch, ForceLoss(), feature=feature_combiner,
                  priors=[bond_potential])

    # Next, we forward the random protein data through the model
    energy, forces = model.forward(coords_torch,
                                   embedding_property=embedding_property)

    # Test if manual and CGnet calculations match numerically
    np.testing.assert_array_equal(energy.detach().numpy(),
                                  manual_energy.detach().numpy())
    np.testing.assert_array_equal(forces.detach().numpy(),
                                  manual_forces.detach().numpy())
