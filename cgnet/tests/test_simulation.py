# Authors: Brooke Husic

import numpy as np
import torch.nn as nn

from cgnet.feature import MoleculeDataset, LinearLayer
from cgnet.network import CGnet, ForceLoss, Simulation
from nose.tools import assert_raises

# Here we create testing data from a random linear protein
# with a random number of frames
frames = np.random.randint(5, 10)
beads = np.random.randint(4, 10)
dims = 3

coords = np.random.randn(frames, beads, dims).astype('float32')
forces = np.random.randn(frames, beads, dims).astype('float32')

dataset = MoleculeDataset(coords, forces)

# Here we construct a single hidden layer architecture with random
# widths and a terminal contraction to a scalar output
arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))

# Here we construct a CGnet model using the above architecture
# as well as variables to be used in CG simulation tests
model = CGnet(arch, ForceLoss()).float()
model.eval()
length = np.random.choice([2, 4])*2  # Number of frames to simulate
save = np.random.choice([2, 4])  # Frequency with which to save simulation
# frames (choice of 2 or 4)


def test_regular_simulation_shape():
    # Test shape of simulation without saving the forces or the potential

    # Grab intitial coordinates as a simulation starting configuration
    # from the moleular dataset
    initial_coordinates = dataset[:][0].reshape(-1, beads, dims)
    sim_length = np.random.choice([2, 4])*2  # Simulation length
    # Frequency of frame saving (either 2 or 4)
    save_interval = np.random.choice([2, 4])

    # Here, we generate the simulation
    model.eval()
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
    model.eval()
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
    model.eval()
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
    model.eval()
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
    model.eval()
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
