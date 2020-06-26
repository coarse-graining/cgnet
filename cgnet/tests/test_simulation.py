# Authors: Brooke Husic
# Contributors: Andreas Kraemer

import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import copy

from cgnet.feature import MoleculeDataset, LinearLayer
from cgnet.network import (CGnet, ForceLoss, Simulation,
                           MultiModelSimulation)
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
# Frequency with which to save simulation
save_interval = np.random.choice([2, 4])
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
# Langevin simulations. Checks are routinely made to make sure that.        #
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The following tests are functional tests based on simulations on a  #
# harmonic potential                                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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
        self.training = False  # needed for cgnet compatibility
        self.feature = None  # needed for cgnet compatibility

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
        potential = (1.0/self.k) * positions**2
        return potential, forces


def test_harmonic_potential_shape_and_temperature():
    # Tests a single harmonic potential simulation for shape and temperature
    # - Tests shapes of trajectory and kinetic energies
    # - Tests that average temperature is about 300
    # - Tests that the standard deviation is less than 10; this is just
    #   heuristic based on trying out the data in a notebook when
    #   writing the test

    # set up model, internal coords, and sim using class attirbutes
    model = HarmonicPotential(k=1, T=300, n_particles=1000, dt=0.001,
                              friction=100, n_sims=1, sim_length=500)

    initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

    my_sim = Simulation(model, initial_coordinates, embeddings=None,
                        beta=model.beta, length=model.sim_length,
                        friction=model.friction, dt=model.dt,
                        masses=model.masses, save_interval=model.save_interval
                        )

    traj = my_sim.simulate()

    # check shape of trajectory and kinetic energies
    assert traj.shape == (model.n_sims, model.sim_length, model.n_particles, 3)
    assert my_sim.kinetic_energies.shape == (1, model.sim_length)

    # Calculate temperatures, removing the first 20 time points (this is
    # about what it takes to get to a constant temperature)
    n_dofs = 3 * model.n_particles
    temperatures = my_sim.kinetic_energies * 2 / n_dofs / model.kB
    temperatures = temperatures[:, 20:]
    mean_temps = np.mean(temperatures, axis=1)

    # Test that the means are all about the right temperature
    np.testing.assert_allclose(np.mean(temperatures, axis=1),
                               np.repeat(model.T, model.n_sims),
                               rtol=1)

    # Test that the stdevs are all less than 25 (heuristic)
    np.testing.assert_array_less(np.std(temperatures, axis=1),
                                 np.repeat(10, model.n_sims))


def test_harmonic_potential_several_temperatures():
    # Tests several harmonic potential simulations for correct temperature.
    # The standard deviation in measured temperature across the simulation
    # is expected to increase as the temperature increases. Heursitically I
    # observed it doesn't tend to exceed a standard deviation of 30 for
    # simulation lengths of 500 and max temperatures of 900.

    temp_parameter = [100, 300, 500, 700, 900]
    mean_temp_measured = []
    std_temp_measured = []

    for temp in temp_parameter:

        # set up model, internal coords, and sim using class attirbutes
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
        sim_temps = my_sim.kinetic_energies * 2 / n_dofs / model.kB
        sim_temps = sim_temps[:, 20:]

        # store mean
        sim_temp_mean = np.mean(sim_temps, axis=1)[0]  # only one simulation
        mean_temp_measured.append(sim_temp_mean)

        # store stdev
        sim_temp_std = np.std(sim_temps, axis=1)[0]  # only one simulation
        std_temp_measured.append(sim_temp_std)

    # Test that the means are all about the right temperature
    np.testing.assert_allclose(temp_parameter,
                               mean_temp_measured,
                               rtol=1)

    # Test that the stdevs are all less than 25 (heuristic)
    np.testing.assert_array_less(std_temp_measured,
                                 np.repeat(30, len(temp_parameter)))

    # Test that the stdevs go up as the temperature goes up
    np.testing.assert_array_equal(std_temp_measured,
                                  sorted(std_temp_measured))


def test_harmonic_potential_zero_friction():
    # Test that zero friction returns a traj of zeroes and kinetic energy
    # of zeroes. Zero friction means vscale will be zero, which means
    # that velocities will only ever be updated by zero. We therefore
    # expect zero friction to leave our velocities completely unchanged.
    # Particularly, in the case where the starting velocities are zero,
    # we expect them to remain zero, and thus the positions do not
    # change either.

    # set up model, internal coords, and sim using class attirbutes
    model = HarmonicPotential(k=1, T=300, n_particles=1000, dt=0.001,
                              friction=0, n_sims=1, sim_length=500)

    initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

    my_sim = Simulation(model, initial_coordinates, embeddings=None,
                        beta=model.beta, length=model.sim_length,
                        friction=model.friction, dt=model.dt,
                        masses=model.masses, save_interval=model.save_interval
                        )

    traj = my_sim.simulate()

    np.testing.assert_array_equal(traj, np.zeros(traj.shape))
    np.testing.assert_array_equal(my_sim.kinetic_energies,
                                  np.zeros(my_sim.kinetic_energies.shape))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The following tests use temporary directories to test the functionality #
# of exporting .npy files and recording logs as the simulation progresses #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def test_saving_numpy_coordinates():
    # Tests, using a temporary directory, the saving of *coordinates*
    # from a Brownian (overdamped Langevin) simulation
    # (i)   That the number of numpy files saved is correct
    # (ii)  That the saved numpy files have the proper shapes
    # (iii) That the contatenation of the saved numpy files are equal to the
    #        trajectory output from the simulation
    n_sims = np.random.randint(1, high=5)
    sim_length = np.random.choice([24, 36])
    npy_interval = np.random.choice([6, 12])
    save_interval = np.random.choice([2, 3])

    n_expected_files = sim_length / npy_interval

    model = HarmonicPotential(k=1, T=300, n_particles=10,
                              dt=0.001, friction=None,
                              n_sims=n_sims, sim_length=sim_length,
                              save_interval=save_interval)

    initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

    with tempfile.TemporaryDirectory() as tmp:
        my_sim = Simulation(model, initial_coordinates, embeddings=None,
                            beta=model.beta, length=model.sim_length,
                            friction=model.friction, dt=model.dt,
                            save_forces=False, save_potential=False,
                            save_interval=model.save_interval,
                            export_interval=npy_interval, filename=tmp+'/test')

        traj = my_sim.simulate()
        assert traj.shape[1] == sim_length / save_interval
        file_list = os.listdir(tmp)

        assert len(file_list) == n_expected_files

        expected_chunk_length = npy_interval / save_interval
        running_traj = None  # needed for (iii)
        for i in range(len(file_list)):
            temp_traj = np.load(tmp+'/'+file_list[i])
            # Test (ii)
            np.testing.assert_array_equal(temp_traj.shape,
                                          [n_sims, expected_chunk_length, model.n_particles, 3])

            if running_traj is None:
                running_traj = temp_traj
            else:
                running_traj = np.concatenate(
                    [running_traj, temp_traj], axis=1)

        # Test (iii)
        np.testing.assert_array_equal(traj, running_traj)


def test_saving_all_quantities():
    # Tests, using a temporary directory, the saving of coordinates,
    # forces, potential, and kinetic energies from a Langevin simulation
    # (i)   That the number of numpy files saved is correct
    # (ii)  That the saved numpy files have the proper shapes
    # (iii) That the contatenation of the saved numpy files are equal to the
    #        trajectory output from the simulation
    n_sims = np.random.randint(1, high=5)
    sim_length = np.random.choice([24, 36])
    npy_interval = np.random.choice([6, 12])
    save_interval = np.random.choice([2, 3])

    n_expected_files = sim_length / npy_interval

    model = HarmonicPotential(k=1, T=300, n_particles=10,
                              dt=0.001, friction=10,
                              n_sims=n_sims, sim_length=sim_length,
                              save_interval=save_interval)

    initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

    with tempfile.TemporaryDirectory() as tmp:
        my_sim = Simulation(model, initial_coordinates, embeddings=None,
                            beta=model.beta, length=model.sim_length,
                            friction=model.friction, dt=model.dt,
                            save_forces=True, save_potential=True,
                            masses=model.masses,
                            save_interval=model.save_interval,
                            export_interval=npy_interval, filename=tmp+'/test')

        traj = my_sim.simulate()
        assert traj.shape[1] == sim_length / save_interval
        file_list = os.listdir(tmp)

        assert len(file_list) == n_expected_files * 4  # coords, forces, pot, ke
        coords_file_list = sorted(
            [file for file in file_list if 'coords' in file])
        force_file_list = sorted(
            [file for file in file_list if 'forces' in file])
        potential_file_list = sorted(
            [file for file in file_list if 'potential' in file])
        ke_file_list = sorted(
            [file for file in file_list if 'kineticenergy' in file])
        file_list_list = [coords_file_list, force_file_list,
                          potential_file_list, ke_file_list]
        expected_chunk_length = npy_interval / save_interval

        # needed for (iii)
        running_coords = None
        running_forces = None
        running_potential = None
        running_ke = None
        running_list = [running_coords, running_forces,
                        running_potential, running_ke]

        obs_list = [my_sim.simulated_coords, my_sim.simulated_forces,
                    my_sim.simulated_potential, my_sim.kinetic_energies]

        for j, obs_file_list in enumerate(file_list_list):
            for i in range(len(obs_file_list)):
                temp_traj = np.load(tmp+'/'+obs_file_list[i])
                # Test (ii)
                if j < 3:
                    np.testing.assert_array_equal(temp_traj.shape,
                                                  [n_sims, expected_chunk_length, model.n_particles, 3])
                else:
                    np.testing.assert_array_equal(temp_traj.shape,
                                                  [n_sims, expected_chunk_length])

                if running_list[j] is None:
                    running_list[j] = temp_traj
                else:
                    running_list[j] = np.concatenate(
                        [running_list[j], temp_traj], axis=1)

            # Test (iii)
            np.testing.assert_array_equal(obs_list[j], running_list[j])


def test_log_file_basics():
    # Tests whether the log file exists, is named correctly, and has the
    # correct number of lines

    n_sims = np.random.randint(1, high=5)
    sim_length = np.random.choice([24, 36])
    log_interval = np.random.choice([6, 12])
    save_interval = np.random.choice([2, 3])

    n_expected_logs = sim_length / log_interval

    model = HarmonicPotential(k=1, T=300, n_particles=10,
                              dt=0.001, friction=None,
                              n_sims=n_sims, sim_length=sim_length,
                              save_interval=save_interval)

    initial_coordinates = torch.zeros((model.n_sims, model.n_particles, 3))

    with tempfile.TemporaryDirectory() as tmp:
        my_sim = Simulation(model, initial_coordinates, embeddings=None,
                            beta=model.beta, length=model.sim_length,
                            friction=model.friction, dt=model.dt,
                            save_forces=False, save_potential=False,
                            save_interval=model.save_interval,
                            log_interval=log_interval, log_type='write',
                            filename=tmp+'/test')

        traj = my_sim.simulate()
        assert traj.shape[1] == sim_length / save_interval
        file_list = os.listdir(tmp)

        # Check that one file exists in the temp directory
        assert len(file_list) == 1

        # Check that it has the proper name
        assert file_list[0] == 'test_log.txt'

        # Gather its lines
        with open(tmp+'/'+file_list[0]) as f:
            line_list = f.readlines()

    # We expect the log file to contain the expected number of logs, plus two
    # extra lines: one at the start and one at the end.
    assert len(line_list) == n_expected_logs + 2


def test_multi_model_simulation():
    # Tests to make sure that forces and potentials are accurately averaged
    # when more than one model is used for a simulation

    # We make 3 to ten random harmonic trap models with randomly chosen 
    # curvature constants
    num_models = np.random.randint(low=3, high=11)
    constants = np.random.uniform(low=1, high=11, size=5)

    # We use the same, random number of sims/particles for all models
    n_sims = np.random.randint(low=10, high=101)
    n_particles = np.random.randint(low=10, high=101)
    masses = n_particles * [np.random.randint(low=1, high=5)]

    models = [HarmonicPotential(k=k, T=300, n_particles=n_particles,
                              dt=0.001, friction=10, n_sims=n_sims,
                              sim_length=10) for k in constants]

    # Here we generate random initial coordinates
    initial_coordinates = torch.randn((n_sims, n_particles, 3))

    my_sim = MultiModelSimulation(models, initial_coordinates,
                                  embeddings=None, length=10,
                                  save_interval=1, masses=masses,
                                  friction=10, dt=0.001)

    # Here we use the 'calculate_potential_and_forces' method from
    # MultiModelSimulation
    avg_potential, avg_forces = my_sim.calculate_potential_and_forces(
                                    initial_coordinates)

    # Next, we compute the average potential and forces manually

    manual_avg_potential = []
    manual_avg_forces = []
    for model in models:
        potential, forces = model(initial_coordinates)
        manual_avg_potential.append(potential)
        manual_avg_forces.append(forces)

    manual_avg_potential = torch.mean(torch.stack(manual_avg_potential), dim=1)
    manual_avg_forces = torch.mean(torch.stack(manual_avg_forces), dim=1)

    # Test to see if the averages calulated by MultiModelSimulation
    # match the averages calculate manually

    np.testing.assert_array_equal(manual_avg_potential.numpy(),
                                  avg_potential.numpy())
    np.testing.assert_array_equal(manual_avg_forces.numpy(),
                                  avg_forces.numpy())


def test_single_model_simulation():
    # Tests to make sure that Simulation and MultiModelSimulation return
    # the same simulation results (coordinates, forces, potential energy,
    # and kinetic energies) if a single model is used in both cases.

    # First, we generate a random integer seed
    seed = np.random.randint(0, 1e6)

    # Next, we set up a model and produce a deep copy
    dt = 0.001
    friction = 10
    k = np.random.randint(1,6)
    n_particles = np.random.randint(1,101)
    n_sims = np.random.randint(1,11)
    initial_coordinates = torch.randn((n_sims, n_particles, 3))
    masses = n_particles * [np.random.randint(low=1, high=5)]
    sim_length = np.random.randint(2,11)
    model = HarmonicPotential(k=k, T=300, n_particles=n_particles,
                              dt=dt, friction=friction, n_sims=n_sims,
                              sim_length=sim_length)
    model_copy = copy.copy(model)
    # Next, we simulate both models. We wrap both of the simulations in
    # a temporary directory as to not generate permanent simulation files
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(model, initial_coordinates, embeddings=None,
                         length=sim_length, save_interval=1, masses=masses,
                         dt=dt, save_forces=True,
                         friction=friction, random_seed=seed,
                         filename=tmp+'/test')
        multi_sim = MultiModelSimulation([model_copy], initial_coordinates,
                         embeddings=None, length=sim_length, save_interval=1,
                         masses=masses, dt=dt, save_forces=True,
                         friction=friction,
                         random_seed=seed, filename=tmp+'/test_copy')
        trajectory = sim.simulate()
        trajectory_copy = multi_sim.simulate()

        # Here, we test the equality of the two simulation results
        print(seed)
        print(trajectory.shape, trajectory_copy.shape)
        np.save("trajectory.npy", trajectory),
        np.save("trajectory_copy.npy", trajectory_copy)
        #assert trajectory.shape == trajectory_copy.shape
        #print(sim.simulated_potential.shape, multi_sim.simulated_potential.shape)
        #assert sim.simulated_forces.shape == multi_sim.simulated_forces.shape
        print(sim.simulated_forces.shape, multi_sim.simulated_forces.shape)
        #assert sim.simulated_potential.shape == multi_sim.simulated_potential.shape
        print(sim.kinetic_energies.shape, multi_sim.kinetic_energies.shape)
        #assert sim.kinetic_energies.shape == multi_sim.kinetic_energies.shape

        np.testing.assert_array_equal(trajectory, trajectory_copy)
        np.testing.assert_array_equal(sim.simulated_potential,
                                      multi_sim.simulated_potential)
        np.testing.assert_array_equal(sim.simulated_forces,
                                      multi_sim.simulated_forces)
        np.testing.assert_array_equal(sim.kinetic_energies,
                                      multi_sim.kinetic_energies)

