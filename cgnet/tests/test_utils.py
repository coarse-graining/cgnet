# Authors: Nick Charron, Brooke Husic

import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from cgnet.network import lipschitz_projection, dataset_loss, Simulation
from cgnet.network import CGnet, ForceLoss
from cgnet.feature import MoleculeDataset, LinearLayer

# Here we create testing data from a random linear protein
# with a random number of frames
frames = np.random.randint(1, 3)
beads = np.random.randint(4, 10)
dims = 3

coords = np.random.randn(frames, beads, dims).astype('float32')
forces = np.random.randn(frames, beads, dims).astype('float32')

# Here, we instance a molecular dataset with sampler and dataloader
dataset = MoleculeDataset(coords, forces)
sampler = SubsetRandomSampler(np.arange(0, frames, 1))
loader = DataLoader(dataset, sampler=sampler,
                    batch_size=np.random.randint(2, high=10))

# Here we construct a single hidden layer architecture with random
# widths and a terminal contraction to a scalar output
arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))

# Here we construct a CGnet model using the above architecture
# as well as variables to be used in CG simulation tests
model = CGnet(arch, ForceLoss()).float()
length = np.random.choice([2, 4])*2  # Number of frames to simulate
save = np.random.choice([2, 4])  # Frequency with which to save simulation
                                 # frames (choice of 2 or 4)


def test_lipschitz_weak_and_strong():
    # Test proper functioning of strong lipschitz projection ( _lambda << 1 )
    # Strongly projected weights should have greatly reduced magnitudes

    # Here we create a single layer test architecture and use it to
    # construct a simple CGnet model. We use a random hidden layer width
    width = np.random.randint(10, high=20)
    test_arch = (LinearLayer(1, width, activation=nn.Tanh()) +
                 LinearLayer(width, 1, activation=None))
    test_model = CGnet(test_arch, ForceLoss()).float()

    # Here we set the lipschitz projection to be extremely strong ( _lambda << 1 )
    _lambda = float(1e-12)

    # We save the numerical values of the pre-projection weights, perform the
    # strong lipschitz projection, and verify that the post-projection weights
    # are greatly reduced in magnitude.
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    lipschitz_projection(test_model, _lambda)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                               if isinstance(layer, nn.Linear)]
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal, pre, post)
        assert np.linalg.norm(pre) > np.linalg.norm(post)

    # Next, we test weak lipschitz projection ( _lambda >> 1 )
    # A weak Lipschitz projection should leave weights entirely unchanged
    # This test is identical to the one above, except here we verify that
    # the post-projection weights are unchanged by the lipshchitz projection
    _lambda = float(1e12)
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    lipschitz_projection(test_model, _lambda)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                               if isinstance(layer, nn.Linear)]
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_array_equal(pre, post)


def test_lipschitz_mask():
    # Test lipschitz mask functionality for random binary mask
    # Using strong Lipschitz projection ( _lambda << 1 )
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
    _lambda = float(1e-12)
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]

    # Next, we create a random binary lipschitz mask, which prevents lipschitz
    # projection for certain random layers
    lip_mask = [np.random.randint(2) for _ in test_arch
                if isinstance(_, nn.Linear)]
    lipschitz_projection(test_model, _lambda, mask=lip_mask)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                               if isinstance(layer, nn.Linear)]
    # Here we verify that the masked layers remain unaffected by the strong
    # Lipschitz projection
    for mask_element, pre, post in zip(lip_mask, pre_projection_weights,
                                       post_projection_weights):
        if mask_element:
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal, pre, post)
            assert np.linalg.norm(pre) > np.linalg.norm(post)
        if not mask_element:
            np.testing.assert_array_equal(pre, post)


def test_dataset_loss():
    # Test dataset loss by comparing results from different batch sizes
    # The loss calculated over the entire dataset should not be affected
    # by the batch size used by the dataloader

    # First, we get dataset loss using the greater-than-one batch size
    # loader from the preamble
    loss = dataset_loss(model, loader)

    # Next, we do the same but use a loader with a batch size of 1
    loader2 = DataLoader(dataset, sampler=sampler, batch_size=1)
    loss2 = dataset_loss(model, loader2)

    # Here, we verify that the two losses over the dataset are equal
    np.testing.assert_allclose(loss, loss2, rtol=1e-5)


def test_regular_simulation_shape():
    # Test shape of simulation without saving the forces or the potential

    # Grab intitial coordinates as a simulation starting configuration
    # from the moleular dataset
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    sim_length = np.random.choice([2, 4])*2  # Simulation length
    # Frequency of frame saving (either 2 or 4)
    save_interval = np.random.choice([2, 4])

    # Here, we generate the simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the potential and the forces are not saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential is None


def test_simulation_saved_forces_shape():
    # Test shape of simulation with only forces saved
    # Grab intitial coordinates as a simulation starting configuration
    # from the moleular dataset
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)

    # Here, we generate the simulation
    my_sim = Simulation(model, initial_coordinates, length=length,
                        save_interval=save, save_forces=True)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the forces, but not the potential, is saved
    assert traj.shape == (frames, length // save, beads, dims)
    assert my_sim.simulated_forces.shape == (
        frames, length // save, beads, dims)
    assert my_sim.simulated_potential is None


def test_simulation_saved_potential_shape():
    # Test shape of simulation with both forces and potential saved
    # Grab intitial coordinates as a simulation starting configuration
    # from the moleular dataset
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    my_sim = Simulation(model, initial_coordinates, length=length,
                        save_interval=save, save_potential=True)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the forces and the potential are saved
    assert traj.shape == (frames, length // save, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential.shape == (
        frames, length // save, beads, 1)


def test_simulation_seeding():
    # Test determinism of simulation with random seed
    # If the same seed is used for two separate simulations,
    # the results (trajectory, forces, potential) should be identical

    # Grab intitial coordinates as a simulation starting configuration
    # from the moleular dataset
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    seed = np.random.randint(1000)  # Save random seed for simulations

    # Generate simulation number one
    sim1 = Simulation(model, initial_coordinates, length=length,
                      save_interval=save, save_forces=True,
                      save_potential=True, random_seed=seed)
    traj1 = sim1.simulate()

    # Generate simulation umber two
    sim2 = Simulation(model, initial_coordinates, length=length,
                      save_interval=save, save_forces=True,
                      save_potential=True, random_seed=seed)
    traj2 = sim2.simulate()

    # Verify that all components of each simulation are equal.
    np.testing.assert_array_equal(traj1, traj2)
    np.testing.assert_array_equal(sim1.simulated_forces, sim2.simulated_forces)
    np.testing.assert_array_equal(sim1.simulated_potential,
                                  sim2.simulated_potential)


def test_simulation_safety():
    # Test whether the simulation indeed will refuse to overwrite
    # existing data unless overwrite is set to true
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)

    # Generate simulation
    sim = Simulation(model, initial_coordinates, length=length,
                     save_interval=save)
    # Check that no simulation is stored
    assert not sim._simulated

    traj = sim.simulate()
    # Check that a simulation is stored now
    assert sim._simulated

    # Check that it can't be overwritten by default
    np.testing.assert_raises(RuntimeError, sim.simulate)

    # Check that it can be overwritten with overwrite=True; i.e. that
    # this command raises no error
    traj2 = sim.simulate(overwrite=True)
    assert sim._simulated
