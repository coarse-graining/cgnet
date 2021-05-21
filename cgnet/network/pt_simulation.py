# Author: Yaoyi Chen to support parallel tempering
# Loosely based on the code from noegroup/reform project
# Revision: fixed bugs in the velocity scaling (Apr 12, 2021)
# Original authors: Brooke Husic, Nick Charron, Jiang Wang
# Original contributors: Dominik Lemm, Andreas Kraemer 

import torch
import numpy as np

import os
import time
import warnings

from cgnet.feature import SchnetFeature
from cgnet.network import Simulation


__all__ = ["calc_beta_from_temperature", "PTSimulation", "PTMultiModelSimulation"]


def calc_beta_from_temperature(temp):
    """Converts a single or a list of temperature(s) in Kelvin 
    to inverse temperature(s) in mol/kcal."""
    from cgnet.feature import KBOLTZMANN, AVOGADRO, JPERKCAL
    return JPERKCAL/KBOLTZMANN/AVOGADRO/np.array(temp)


class PTSimulation(Simulation):
    """Simulate an artificial trajectory from a CGnet with parallel tempering.
    For thoeretical details on (overdamped) Langevin integration schemes,
    see help(cgnet.network.Simulation).
    For thoeretical details on replica exchange/parallel tempering, see 
    https://github.com/noegroup/reform.
    Note that currently we only implement parallel tempering for Langevin 
    dynamics, so please make sure you provide correct parameters for that.
    Be aware that the output will contain information (e.g., coordinates) 
    for all replicas. 

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        Trained model used to generate simulation data
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Unit should be ps.
    betas : list of floats (default=[1.0])
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
        This parameter will determine how many replicas will be simulated and
        at which temeperatures.
    friction : float (default=None)
        Please provide a float friction value for Langevin dynamics.
    masses : list of floats (default=None)
        Must be a list of floats where the float at mass index i corresponds 
        to the ith CG bead. (Note: please divide it by 418.4 if you are using
        force unit kcal/mol/A and time .)
    diffusion : float (default=1.0)
        Not used for Langevin dynamics
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    exchange_interval : int (default=100)
        The interval at which we attempt to exchange the replicas at different 
        temperatures. Must be a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.

    Notes
    -----
    For notes from the original code, see help(cgnet.network.Simulation).

    """

    def __init__(self, model, initial_coordinates, betas=[1.0], 
                 exchange_interval=100, **kwargs):
        # the basic idea is to reuse all routines in the original code to
        # perform (n_replicas * n_sims) independent simulations of the same
        # system and to attempt replica exchange among corresponding replicas
        # at the given time interval.
        # Example:
        # input betas: [1.0, 0.68, 0.5]
        # input initial_coordinates: [conf_0, conf_1] (each conf_i has shape
        #                                              [n_beads, n_dims])
        # actual internal n_sims = 3 * 2 = 6
        # betas for simulation: [1.0, 1.0, 0.68, 0.68, 0.5, 0.5]
        # actual internal initial_coordinates = [conf_0, conf_1, conf_0,
        #                                        conf_1, conf_0, conf_1]

        # make sure the simulation will be in Langevin scheme
        if kwargs.get("friction") is None:  # overdamped langevin
            raise ValueError('Please provide a valid friction value, '
                             'since we current only support Langevin '
                             'dynamics.')

        # checking customized inputs
        betas = np.array(betas)
        if len(betas.shape) != 1 or betas.shape[0] <= 1:
            raise ValueError('betas must have shape (n_replicas,), where '
                             'n_replicas > 1.')
        self._betas = betas
        if type(exchange_interval) is not int or exchange_interval < 0:
            raise ValueError('exchange_interval must be a positive integer.')
        self.exchange_interval = exchange_interval

        # identify number of replicas
        self.n_replicas = len(self._betas)

        # preparing initial coordinates for each replica
        if type(initial_coordinates) is torch.Tensor:
            initial_coordinates = initial_coordinates.detach().cpu().numpy()
        new_initial_coordinates = np.concatenate([initial_coordinates] * 
                                                 self.n_replicas)
        
        # preparing embeddings for each replica
        embeddings = kwargs.get("embeddings")
        if embeddings is not None:
            if type(embeddings) is torch.Tensor:
                embeddings = embeddings.detach().cpu().numpy()
            embeddings = np.concatenate([embeddings] * self.n_replicas)
            embeddings = torch.tensor(embeddings, dtype=torch.int64
                                     ).to(kwargs.get("device"))
        kwargs["embeddings"] = embeddings

        # original initialization (note that we don't use self.beta any more)
        super(PTSimulation, self).__init__(model,
                                           torch.tensor(new_initial_coordinates),
                                           **kwargs)

        # set up betas for simulation
        self._n_indep = len(initial_coordinates)
        self._betas_x = np.repeat(self._betas, self._n_indep)
        self._betas_for_simulation = torch.tensor(self._betas_x[:, None, None],
                                                  dtype=torch.float32
                                                 ).to(self.device)
        # for replica exchange pair proposing
        self._propose_even_pairs = True
        even_pairs = [(i, i + 1) for i in np.arange(self.n_replicas)[:-1:2]]
        odd_pairs = [(i, i + 1) for i in np.arange(self.n_replicas)[1:-1:2]]
        if len(odd_pairs) == 0:
            odd_pairs = even_pairs
        pair_a = []
        pair_b = []
        for pair in even_pairs:
            pair_a.append(np.arange(self._n_indep) + pair[0] * self._n_indep)
            pair_b.append(np.arange(self._n_indep) + pair[1] * self._n_indep)
        self._even_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]
        pair_a = []
        pair_b = []
        for pair in odd_pairs:
            pair_a.append(np.arange(self._n_indep) + pair[0] * self._n_indep)
            pair_b.append(np.arange(self._n_indep) + pair[1] * self._n_indep)
        self._odd_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]


    def get_replica_info(self, replica_num=0):
        if type(replica_num) is not int or replica_num < 0 or \
           replica_num >= self.n_replicas:
            raise ValueError('Please provide a valid replica number.')
        indices = np.arange(replica_num * self._n_indep,
                            (replica_num + 1) * self._n_indep)
        return {"beta": self._betas[replica_num], 
                "indices_in_the_output": indices}


    def _langevin_timestep(self, x_old, v_old, forces):
        """Heavy lifter for Langevin dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : torch.Tensor
            velocities before propagation
        forces: torch.Tensor
            forces at x_old
        """

        # BB (velocity update); uses whole timestep
        v_new = v_old + self.dt * forces / self.masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt / 2.

        # O (noise)
        noise = torch.sqrt(1. / self._betas_for_simulation / self.masses[:, None])
        noise = noise * torch.randn(size=x_new.size(),
                                    generator=self.rng).to(self.device)
        v_new = v_new * self.vscale
        v_new = v_new + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt / 2.

        return x_new, v_new


    def _overdamped_timestep(self, x_old, v_old, forces):
        """Heavy lifter for overdamped Langevin (Brownian) dynamics

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates before propagataion
        v_old : None
            Placeholder
        forces: torch.Tensor
            forces at x_old
        """
        raise NotImplementedError()


    def _get_proposed_pairs(self):
        """Proposes the even and odd pairs alternatively."""
        if self._propose_even_pairs:
            self._propose_even_pairs = False
            return self._even_pairs
        else:
            self._propose_even_pairs = True
            return self._odd_pairs


    def _detect_exchange(self, potentials):
        """Proposes and checks pairs to be exchanged for parallel tempering.
        Modified from `reform`."""
        pair_a, pair_b = self._get_proposed_pairs()
        
        u_a, u_b = potentials[pair_a], potentials[pair_b]
        betas_a, betas_b = self._betas_x[pair_a], self._betas_x[pair_b]
        p_pair = np.exp((u_a - u_b) * (betas_a - betas_b))
        approved = np.random.rand(len(p_pair)) < p_pair
        #print("Exchange rate: %.2f%%" % (approved.sum() / len(pair_a) * 100.))
        self._replica_exchange_attempts += len(pair_a)
        self._replica_exchange_approved += approved.sum()
        pairs_for_exchange = {"a": pair_a[approved], "b": pair_b[approved]}
        return pairs_for_exchange

    def _get_velo_scaling_factors(self, indices_old, indices_new):
        """Velocity scaling factor for Langevin simulation: 
        \sqrt(t_new/t_old)
        """
        return torch.sqrt(self._betas_for_simulation[indices_old] /
                          self._betas_for_simulation[indices_new])

    def _perform_exchange(self, pairs_for_exchange, xs, vs):
        """Exchanges the coordinates and  given pairs"""
        pair_a, pair_b = pairs_for_exchange["a"], pairs_for_exchange["b"]
        # exchange the coordinates
        x_changed = xs.detach().clone()
        x_changed[pair_a, :, :] = xs[pair_b, :, :]
        x_changed[pair_b, :, :] = xs[pair_a, :, :]
        # scale and exchange the velocities
        v_changed = vs.detach().clone()
        v_changed[pair_a, :, :] = vs[pair_b, :, :] * \
                               self._get_velo_scaling_factors(pair_b, pair_a)
        v_changed[pair_b, :, :] = vs[pair_a, :, :] * \
                               self._get_velo_scaling_factors(pair_a, pair_b)
        return x_changed, v_changed

    def simulate(self, overwrite=False):
        """Generates independent simulations.

        Parameters
        ----------
        overwrite : Bool (default=False)
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords : np.ndarray
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval

        Attributes
        ----------
        simulated_forces : np.ndarray or None
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            If simulated_forces is True, stores the simulation forces
            calculated for each frame in the simulation at the
            save_interval
        simulated_potential : np.ndarray or None
            Shape [n_simulations, n_frames, [potential dimensions]]
            If simulated_potential is True, stores the potential calculated
            for each frame in simulation at the save_interval
        kinetic_energies : np.ndarray or None
            If friction is not None, stores the kinetic energy calculated
            for each frame in the simulation at the save_interval

        """
        self._set_up_simulation(overwrite)

        # counters for replica exchange
        self._replica_exchange_attempts = 0
        self._replica_exchange_approved = 0

        if self.log_interval is not None:
            printstring = ("Generating {} sets of independent parallel-"
                           "tempering simulations at {} different temperatures"
                           " of length {} saved at {}-step intervals ({})"
                          ).format(self._n_indep, self.n_replicas, 
                                   self.length, self.save_interval,
                                   time.asctime())
            printstring += ("\nThere will be {} = {} * {} trajectories "
                            "recorded.").format(self.n_sims, self._n_indep, 
                                                self.n_replicas)
            if self.log_type == 'print':
                print(printstring)

            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        x_old = self._initial_x

        # for each simulation step
        if self.friction is None:
            v_old = None
        else:
            # initialize velocities at zero
            v_old = torch.tensor(np.zeros(x_old.shape),
                                 dtype=torch.float32).to(self.device)
            # v_old = v_old + torch.randn(size=v_old.size(),
            #                             generator=self.rng).to(self.device)

        for t in range(self.length):
            # produce potential and forces from model
            potential, forces = self.calculate_potential_and_forces(x_old)
            potential = potential.detach()
            forces = forces.detach()

            # step forward in time
            with torch.no_grad():
                x_new, v_new = self._timestep(x_old, v_old, forces)

            # save to arrays if relevant
            if (t+1) % self.save_interval == 0:
                self._save_timepoint(x_new, v_new, forces, potential, t)

                # save numpys if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self._save_numpy((t+1) // self.save_interval)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self._log_progress((t+1) // self.save_interval)

            # !!! attempt to exchange !!!
            if (t+1) % self.exchange_interval == 0:
                # get potentials
                x_new = x_new.detach().requires_grad_(True).to(self.device)
                potential_new, _ = self.calculate_potential_and_forces(x_new)
                potential_new = potential_new.detach().cpu().numpy()[:, 0]
                pairs_for_exchange = self._detect_exchange(potential_new)
                x_new, v_new = self._perform_exchange(pairs_for_exchange,
                                                      x_new, v_new)
                
            # prepare for next timestep
            x_old = x_new.detach().requires_grad_(True).to(self.device)
            v_old = v_new

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t+1) % self.export_interval > 0:
                self._save_numpy(t+1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            attempted = self._replica_exchange_attempts
            exchanged = self._replica_exchange_approved
            printstring = 'Done simulating ({})'.format(time.asctime())
            printstring += "\nReplica-exchange rate: %.2f%% (%d/%d)" % (
                            exchanged / attempted * 100., exchanged,
                            attempted)
            printstring += ("\nNote that you can call .get_replica_info"
                            "(#replica) to query the inverse temperature"
                            " and trajectory indices for a given replica.")
            if self.log_type == 'print':
                print(printstring)
            elif self.log_type == 'write':
                printstring += '\n'
                file = open(self._log_file, 'a')
                file.write(printstring)
                file.close()

        # reshape output attributes
        self.simulated_coords = self._swap_and_export(
            self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(
                self.simulated_forces)

        if self.save_potential:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential)

        if self.friction is not None:
            self.kinetic_energies = self._swap_and_export(
                self.kinetic_energies)

        self._simulated = True

        return self.simulated_coords


class PTMultiModelSimulation(PTSimulation):
    """Simulation that integrates CG coordinates forward in time using
    the average forces predicted from more than one CGnet model with parallel
    tempering.
    See help(cgnet.network.PTSimulation) for details.

    Parameters
    ----------
    models: list of cgnet.network.CGNet() instances
        The list of models from which predicted forces will be averaged over.
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Unit should be ps.
    betas : list of floats (default=[1.0])
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
        This parameter will determine how many replicas will be simulated and
        at which temeperatures.
    friction : float (default=None)
        Please provide a float friction value for Langevin dynamics.
    masses : list of floats (default=None)
        Must be a list of floats where the float at mass index i corresponds 
        to the ith CG bead. (Note: please divide it by 418.4 if you are using
        force unit kcal/mol/A and time .)
    diffusion : float (default=1.0)
        Not used for Langevin dynamics
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    exchange_interval : int (default=100)
        The interval at which we attempt to exchange the replicas at different 
        temperatures. Must be a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.


    Notes
    -----
    The relationship between this class and cgnet.network.PTsimulation is like 
    the one between cgnet.network.PTMultiModelSimulation and 
    cgnet.network.Simulation.
    I just copied the code over to avoid the problem of multiple inheritance.
    """

    def __init__(self, models, initial_coordinates, **kwargs):

        super(PTMultiModelSimulation, self).__init__(models, initial_coordinates,
                                                   **kwargs)

        self._input_model_list_check(models)
        self.models = models

    def _input_model_list_check(self, models):
        for idx, individual_model in enumerate(models):
            self._input_model_checks(individual_model, idx=idx)

    def calculate_potential_and_forces(self, x_old):
        """Method to calculate average predicted potentials and forces from
        forwarding the x_old through each of the models in self.models

        Parameters
        ----------
        x_old : torch.Tensor
            coordinates from the previous timestep

        Returns
        -------
        potential : torch.Tensor
            scalar potential averaged over all the models in self.models
        forces : torch.Tensor
            vector forces averaged over all the models in self.models
        """
        potential_list = []
        forces_list = []
        for model in self.models:
            potential, forces = model(x_old, self.embeddings)
            potential_list.append(potential)
            forces_list.append(forces)
        mean_potential =  sum(potential_list) / len(potential_list)
        mean_forces =  sum(forces_list) / len(forces_list)

        # make sure the mean did not alter the shapes of potential/forces
        # using the last potential/force
        assert mean_potential.size() == potential.size()
        assert mean_forces.size() == forces.size()

        return mean_potential, mean_forces
