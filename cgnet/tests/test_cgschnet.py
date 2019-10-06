# Authors: Brooke Husic
# Contributors: Dominik Lemm, Nick Charron

import numpy as np
import torch

from cgnet.feature import (GeometryFeature, CGBeadEmbedding, SchnetFeature,
                           FeatureCombiner, LinearLayer, ShiftedSoftplus)

from cgnet.network import CGnet, ForceLoss, Simulation


frames = np.random.randint(10, 30)  # Number of frames
beads = np.random.randint(5, 10)  # Number of coarse-granined beads
dims = 3  # Number of dimensions

coords = torch.randn((frames, beads, 3), requires_grad=True)
forces = torch.randn((frames, beads, 3), requires_grad=True)
embeddings = torch.randint(3, size=(coords.shape[:2]))

# hyperparameters
n_embeddings = np.random.randint(5, 10)
embedding_dim = np.random.randint(50, 100)
# The number of interaction blocks must be less than or equal to the
# number of embeddings
n_interaction_blocks = np.random.randint(2, 5)
n_gaussians = np.random.randint(12, 24)
n_layers = np.random.randint(2, 5)


def _make_cgschnet_model(beads, n_embeddings, embedding_dim,
                         n_interaction_blocks, n_gaussians, n_layers):
    geometry_feature = GeometryFeature(n_beads=beads)
    embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings,
                                      embedding_dim=embedding_dim)
    schnet_feature = SchnetFeature(feature_size=embedding_dim,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=n_interaction_blocks,
                                   calculate_geometry=False,
                                   n_beads=beads,
                                   neighbor_cutoff=None,
                                   n_gaussians=n_gaussians)
    layer_list = [geometry_feature, schnet_feature]

    distance_indices = np.arange(beads * (beads-1) // 2)
    feature_combiner = FeatureCombiner(layer_list,
                                       distance_indices=distance_indices)
    layers = LinearLayer(embedding_dim, embedding_dim,
                         activation=ShiftedSoftplus())
    for l in range(n_layers - 1):
        layers += LinearLayer(embedding_dim, embedding_dim,
                              activation=ShiftedSoftplus())

    model = CGnet(layers, ForceLoss(), feature=feature_combiner)

    return model


# Create our cgschnet model
model = _make_cgschnet_model(beads, n_embeddings, embedding_dim,
                             n_interaction_blocks,  n_gaussians, n_layers)


def test_cgschnet_shapes():
    # Test the architecture, output energy, and output force shapes of
    # a CGSchnet model
    energy, force = model.forward(coords, embeddings)

    # The number of layers should be 2*n_layers because each "layer" results
    # in one linear layer and one nonlinearity in the archeticture object.
    assert len(model.arch) == 2*n_layers

    # Test the shapes of the energy and force outputs
    np.testing.assert_array_equal(energy.size(), [frames, embedding_dim])
    np.testing.assert_array_equal(force.size(), [frames, beads, 3])


def test_cgschnet_simulation_shapes():
    # Test simulation with embeddings and make sure the shapes of
    # the simulated coordinates, forces, and potential are correct
    sim_length = np.random.randint(10, 20)
    sim = Simulation(model, coords, embeddings, length=sim_length,
                     save_interval=1, beta=1., save_forces=True,
                     save_potential=True)

    traj = sim.simulate()

    np.testing.assert_array_equal(sim.simulated_traj.shape,
                                  [frames, sim_length, beads, 3])

    np.testing.assert_array_equal(sim.simulated_forces.shape,
                                  [frames, sim_length, beads, 3])

    np.testing.assert_array_equal(sim.simulated_potential.shape,
                                  [frames, sim_length, embedding_dim])
