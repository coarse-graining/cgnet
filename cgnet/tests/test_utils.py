# Authors: Nick Charron, Brooke Husic

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from cgnet.network import lipschitz_projection, dataset_loss, Simulation
from cgnet.network import CGnet, ForceLoss
from cgnet.feature import MoleculeDataset, LinearLayer

frames = np.random.randint(1, 3)
beads = np.random.randint(4, 10)
dims = 2

coords = np.random.randn(frames, beads, dims).astype('float32')
forces = np.random.randn(frames, beads, dims).astype('float32')
dataset = MoleculeDataset(coords, forces)
sampler = SubsetRandomSampler(np.arange(0, frames, 1))
loader = DataLoader(dataset, sampler=sampler,
                    batch_size=np.random.randint(2, high=10))

arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))

model = CGnet(arch, ForceLoss()).float()
lip_mask = [np.random.randint(2) for _ in arch if isinstance(_, nn.Linear)]
length = np.random.choice([2, 4])*2
save = np.random.choice([2, 4])


def test_lipschitz():
    # Test hard lipschitz projection
    test_arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))
    test_model = CGnet(test_arch, ForceLoss()).float()
    _lambda = float(1e-12)
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    lipschitz_projection(test_model, _lambda)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal, pre, post)

    # Test soft lipschitz projection
    _lambda = float(1e12)
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    lipschitz_projection(test_model, _lambda)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_array_equal(pre, post)

def test_lipschitz_mask():
    # Test lipschitz mask functionality
    # use strong lipschitz projection
    test_arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))
    test_model = CGnet(test_arch, ForceLoss()).float()
    _lambda = float(1e-12)
    pre_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]

    lipschitz_projection(test_model, _lambda, mask=lip_mask)
    post_projection_weights = [layer.weight.data for layer in test_model.arch
                              if isinstance(layer, nn.Linear)]
    for mask_element, pre, post in zip(lip_mask, pre_projection_weights,
                                       post_projection_weights):
        if mask_element:
           np.testing.assert_raises(AssertionError,
                                    np.testing.assert_array_equal, pre, post)
        if not mask_element:
           np.testing.assert_array_equal(pre, post)

def test_dataset_loss():
    # Test dataset loss by comparing results from different batch sizes
    # Batch size != 1
    loss = dataset_loss(model, loader)

    # Batch size = 1
    loader2 = DataLoader(dataset, sampler=sampler, batch_size=1)
    loss2 = dataset_loss(model, loader2)

    np.testing.assert_allclose(loss, loss2, rtol=1e-5)


def test_regular_simulation():
    # Test shape of simulation with nothing else saved
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    length = np.random.choice([2, 4])*2
    save = np.random.choice([2, 4])
    my_sim = Simulation(model, initial_coordinates, length=length,
                        save_interval=save)
    traj = my_sim.simulate()

    assert traj.shape == (frames, length // save, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential is None


def test_simulation_saved_forces():
    # Test shape of simulation and forces saved
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    my_sim = Simulation(model, initial_coordinates, length=length,
                        save_interval=save, save_forces=True)
    traj = my_sim.simulate()

    assert traj.shape == (frames, length // save, beads, dims)
    assert my_sim.simulated_forces.shape == (
        frames, length // save, beads, dims)
    assert my_sim.simulated_potential is None


def test_simulation_saved_potential():
    # Test shape of simulation, potential, and forces saved
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    my_sim = Simulation(model, initial_coordinates, length=length,
                        save_interval=save, save_potential=True)
    traj = my_sim.simulate()

    assert traj.shape == (frames, length // save, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential.shape == (
        frames, length // save, beads, 1)


def test_simulation_seeding():
    # Test determinism of simulation with random seed
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    seed = np.random.randint(1000)

    sim1 = Simulation(model, initial_coordinates, length=length,
                      save_interval=save, save_forces=True,
                      save_potential=True, random_seed=seed)
    traj1 = sim1.simulate()

    sim2 = Simulation(model, initial_coordinates, length=length,
                      save_interval=save, save_forces=True,
                      save_potential=True, random_seed=seed)
    traj2 = sim2.simulate()

    np.testing.assert_array_equal(traj1, traj2)
    np.testing.assert_array_equal(sim1.simulated_forces, sim2.simulated_forces)
    np.testing.assert_array_equal(sim1.simulated_potential,
                                  sim2.simulated_potential)
