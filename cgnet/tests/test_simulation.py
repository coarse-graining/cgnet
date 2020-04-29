# Authors: Brooke Husic
# Contributors: Andreas Kraemer

import numpy as np
import torch
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
sim_length = np.random.choice([2, 4])*2  # Number of frames to simulate
save_interval = np.random.choice([2, 4])  # Frequency with which to save simulation
                                 # frames (choice of 2 or 4)

# Grab intitial coordinates as a simulation starting configuration
# from the moleular dataset
initial_coordinates = dataset[:][0].reshape(-1, beads, dims)

# Langevin simulation parameters
masses = np.ones(beads)
friction = np.random.randint(10, 20)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The following tests probe basic shapes/functionalities of the simulation  #
# class and are repeated for Brownian (i.e., overdamped Langevin) and.      #
# Langevin simulations. Checks are routinely made to mkae sure that.        #
# there are no kinetic energies in the former, but there are in the latter. #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_brownian_simulation_shape():
    # Test shape of Brownian (overdamped Langevin) simulation without
    # saving the forces or the potential

    # Generate simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the potential and the forces are not saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential is None
    assert my_sim.kinetic_energies is None


def test_langevin_simulation_shape():
    # Test shape of Lanvegin simulation without saving the forces or the
    # potential

    # Generate simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval, friction=friction,
                        masses=masses)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the potential and the forces are not saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential is None
    assert my_sim.kinetic_energies.shape == (frames,
                                             sim_length // save_interval)


def test_brownian_simulation_saved_forces_shape():
    # Test shape of brownian (overdamped langevin) simulation with only
    # forces saved

    # Generate simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval, save_forces=True)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the forces, but not the potential, is saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces.shape == (
        frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_potential is None
    assert my_sim.kinetic_energies is None


def test_langevin_simulation_saved_forces_shape():
    # Test shape of langevin simulation with only forces saved

    # Generate simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval, save_forces=True,
                        friction=friction, masses=masses)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the forces, but not the potential, is saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces.shape == (
        frames, sim_length // save_interval, beads, dims
        )
    assert my_sim.simulated_potential is None
    assert my_sim.kinetic_energies.shape == (frames,
                                             sim_length // save_interval)


def test_brownian_simulation_saved_potential_shape():
    # Test shape of brownian (overdamped langevin) simulation with both
    # forces and potential saved

    # Generate simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval, save_potential=True)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the forces and the potential are saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential.shape == (
        frames, sim_length // save_interval, beads, 1
        )
    assert my_sim.kinetic_energies is None


def test_langevin_simulation_saved_potential_shape():
    # Test shape of langevin simulation with both forces and potential saved

    # Generate simulation
    my_sim = Simulation(model, initial_coordinates, length=sim_length,
                        save_interval=save_interval, save_potential=True,
                        friction=friction, masses=masses)
    traj = my_sim.simulate()

    # Here, we verify that the trajectory shape corresponds with the
    # choices of simulation length and saving frequency above
    # We also verify that the forces and the potential are saved
    assert traj.shape == (frames, sim_length // save_interval, beads, dims)
    assert my_sim.simulated_forces is None
    assert my_sim.simulated_potential.shape == (
        frames, sim_length // save_interval, beads, 1
        )
    assert my_sim.kinetic_energies.shape == (frames,
                                             sim_length // save_interval)

def test_brownian_simulation_seeding():
    # Test determinism of Brownian (overdamped langevin) simulation with
    # random seed. If the same seed is used for two separate simulations,
    # the results (trajectory, forces, potential) should be identical

    seed = np.random.randint(1000)  # Save random seed for simulations

    # Generate simulation number one
    sim1 = Simulation(model, initial_coordinates, length=sim_length,
                      save_interval=save_interval, save_forces=True,
                      save_potential=True, random_seed=seed)
    traj1 = sim1.simulate()

    # Generate simulation umber two
    sim2 = Simulation(model, initial_coordinates, length=sim_length,
                      save_interval=save_interval, save_forces=True,
                      save_potential=True, random_seed=seed)
    traj2 = sim2.simulate()

    # Verify that all components of each simulation are equal.
    np.testing.assert_array_equal(traj1, traj2)
    np.testing.assert_array_equal(sim1.simulated_forces, sim2.simulated_forces)
    np.testing.assert_array_equal(sim1.simulated_potential,
                                  sim2.simulated_potential)
    assert sim1.kinetic_energies is None
    assert sim2.kinetic_energies is None


def test_langevin_simulation_seeding():
    # Test determinism of Langevin simulation with random seed. If the 
    # same seed is used for two separate simulations, the results
    # (trajectory, forces, potential) should be identical

    seed = np.random.randint(1000)  # Save random seed for simulations

    # Generate simulation number one
    sim1 = Simulation(model, initial_coordinates, length=sim_length,
                      save_interval=save_interval, save_forces=True,
                      save_potential=True, random_seed=seed,
                      friction=friction, masses=masses)
    traj1 = sim1.simulate()

    # Generate simulation umber two
    sim2 = Simulation(model, initial_coordinates, length=sim_length,
                      save_interval=save_interval, save_forces=True,
                      save_potential=True, random_seed=seed,
                      friction=friction, masses=masses)
    traj2 = sim2.simulate()

    # Verify that all components of each simulation are equal, and not
    # because they're None
    assert traj1 is not None
    assert traj2 is not None
    assert sim1.simulated_forces is not None
    assert sim2.simulated_forces is not None
    assert sim1.simulated_potential is not None
    assert sim2.simulated_potential is not None
    assert sim1.kinetic_energies is not None
    assert sim2.kinetic_energies is not None

    np.testing.assert_array_equal(traj1, traj2)
    np.testing.assert_array_equal(sim1.simulated_forces, sim2.simulated_forces)
    np.testing.assert_array_equal(sim1.simulated_potential,
                                  sim2.simulated_potential)
    np.testing.assert_array_equal(sim1.kinetic_energies, sim2.kinetic_energies)


def test_brownian_simulation_safety():
    # Test whether a brownian (overdamped langevin) simulation indeed will
    # refuse to overwrite existing data unless overwrite is set to true

    # Generate simulation
    sim = Simulation(model, initial_coordinates, length=sim_length,
                     save_interval=save_interval)
    # Check that no simulation is stored
    assert not sim._simulated

    traj = sim.simulate()
    assert sim.kinetic_energies is None
    # Check that a simulation is stored now
    assert sim._simulated

    # Check that it can't be overwritten by default
    np.testing.assert_raises(RuntimeError, sim.simulate)

    # Check that it can be overwritten with overwrite=True; i.e. that
    # this command raises no error
    traj2 = sim.simulate(overwrite=True)
    assert sim._simulated


def test_langevin_simulation_safety():
    # Test whether a brownian (overdamped langevin) simulation indeed will
    # refuse to overwrite existing data unless overwrite is set to true

    # Generate simulation
    sim = Simulation(model, initial_coordinates, length=sim_length,
                     save_interval=save_interval, friction=friction,
                     masses=masses)
    # Check that no simulation is stored
    assert not sim._simulated

    traj = sim.simulate()
    assert sim.kinetic_energies is not None
    # Check that a simulation is stored now
    assert sim._simulated

    # Check that it can't be overwritten by default
    np.testing.assert_raises(RuntimeError, sim.simulate)

    # Check that it can be overwritten with overwrite=True; i.e. that
    # this command raises no error
    traj2 = sim.simulate(overwrite=True)
    assert sim._simulated

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ....
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class HarmonicPotential():
    """Defines a harmonic potential according to

    U(x) = k/2 * x**2

    with attributes enabling it to get through all the error checks

    Parameters
    ----------
    k : float (default=1)
        Constant for harmonic potential (see above)
    T : float (defaul=300)
        Temperature in K
    particles : int (default=1000)
        Number of particles in a given simulation
    dt : float (default=0.001)
        Length of simulation timestep
    friction : float (default=100)
        Friction constant (see cgnet.network.Simulation)
    sims : int (default=1)
        Number of simulations to run in parallel
    sim_length : int (default=500)
        Number of time steps per simulation
    save_interval : int (default=1)
        Number of time points at which to record simulation results
    """
    
    def __init__(self, k=1, T=300, n_particles=1000, dt=0.001, friction=100,
                 n_sims=1, sim_length=500, save_interval=1):
        self.k = k
        self.training = False
        self.feature = None

        self.T = T
        self.kB = 0.008314472471220215
        self.beta = 1. / self.kB / self.T

        self.n_particles = n_particles
        self.dt = dt
        self.friction = friction
        self.n_sims = n_sims
        self.sim_length = sim_length
        self.save_interval = save_interval

        self.masses = np.ones((n_particles,))

        
    def __call__(self, positions, embeddings=None):
        """in kilojoule/mole/nm"""
        forces = -self.k * positions
        potential = torch.zeros(1) # dont need meaningful values here
        return potential, forces


def test_harmonic_potential_shape_and_temperature():
    # TODO
    # - Tests shapes of trajectory and kinetic energies
    # - Tests that average temperature is about 300

    model = HarmonicPotential(k=1, T=300, n_particles=1000, dt=0.001,
                              friction=100, n_sims=1, sim_length=500)

    initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

    my_sim = Simulation(model, initial_coordinates, embeddings=None,
                        beta=model.beta, length=model.sim_length,
                        friction=model.friction, dt=model.dt,
                        masses=model.masses, save_interval=model.save_interval
                        )

    traj = my_sim.simulate()
    assert traj.shape == (model.n_sims, model.sim_length, model.n_particles, 3)
    assert my_sim.kinetic_energies.shape == (1, model.sim_length)

    n_dofs = 3 * model.n_particles
    temperatures = my_sim.kinetic_energies * 2 / n_dofs / model.kB
    temperatures = temperatures[:, 20:]
    mean_temps = np.mean(temperatures, axis=1)

    np.testing.assert_allclose(np.mean(temperatures, axis=1),
                               np.repeat(model.T, model.n_sims),               
                               rtol=1)

def test_harmonic_potential_several_temperatures():
    # TODO
    temps = [np.random.randint(low=50, high=900) for _ in range(5)]

    for temp in temps:
        model = HarmonicPotential(k=1, T=temp, n_particles=1000, dt=0.001,
                                  friction=100, n_sims=1, sim_length=500)

        initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

        my_sim = Simulation(model, initial_coordinates, embeddings=None,
                            beta=model.beta, length=model.sim_length,
                            friction=model.friction, dt=model.dt,
                            masses=model.masses, save_interval=model.save_interval
                            )

        traj = my_sim.simulate()
        n_dofs = 3 * model.n_particles
        temperatures = my_sim.kinetic_energies * 2 / n_dofs / model.kB
        temperatures = temperatures[:, 20:]

        np.testing.assert_allclose(np.mean(temperatures, axis=1),
                                   np.repeat(model.T, model.n_sims),               
                                   rtol=1)
