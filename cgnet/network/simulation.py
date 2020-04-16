# Authors: Brooke Husic, Nick Charron, Jiang Wang
# Contributors: Dominik Lemm

import torch
import numpy as np
import warnings

from cgnet.feature import SchnetFeature


class _Simulation():
    """TODO"""
    def __init__(self):
        pass

    def swap_axes(self, data, axis1, axis2):
        """Helper method to exchange the zeroth and first axes of tensors after
        simulations have finished

        Parameters
        ----------
        data : torch.Tensor
            Tensor to perform the axis swtich upon. Size
            [n_timesteps, n_simulations, n_beads, n_dims]
        axis1 : int
            Zero-based index of the first axis to swap
        axis2 : int
            Zero-based index of the second axis to swap

        Returns
        -------
        swapped_data : torch.Tensor
            Axes-swapped tensor. Size
            [n_timesteps, n_simulations, n_beads, n_dims]
        """
        axes = list(range(len(data.size())))
        axes[axis1] = axis2
        axes[axis2] = axis1
        swapped_data = data.permute(*axes)
        return swapped_data

    def simulate(self):
        raise NotImplementedError(
            "simulate method must be implemented on a class-by-class basis"
            )


class Simulation(_Simulation):
    """TODO"""
    def __init__(self, model, initial_coordinates, embeddings=None,
                 save_forces=False, save_potential=False, length=100,
                 save_interval=10, dt=5e-4, diffusion=1.0, beta=1.0,
                 verbose=False, random_seed=None, device=torch.device('cpu')):
        if length % save_interval != 0:
            raise ValueError(
                'The save_interval must be a factor of the simulation length'
            )

        if len(initial_coordinates.shape) != 3:
            raise ValueError(
                'initial_coordinates shape must be [frames, beads, dimensions]'
            )

        if embeddings is None:
            try:
                if np.any([type(model.feature.layer_list[i]) == SchnetFeature
                           for i in range(len(model.feature.layer_list))]):
                    raise RuntimeError('Since you have a SchnetFeature, you must \
                                        provide an embeddings array')
            except:
                if type(model.feature) == SchnetFeature:
                    raise RuntimeError('Since you have a SchnetFeature, you must \
                                        provide an embeddings array')

        if embeddings is not None:
            if len(embeddings.shape) != 2:
                raise ValueError('embeddings shape must be [frames, beads]')

            if initial_coordinates.shape[:2] != embeddings.shape:
                raise ValueError('initial_coordinates and embeddings '
                                 'must have the same first two dimensions')

        if type(initial_coordinates) is not torch.Tensor:
            initial_coordinates = torch.tensor(initial_coordinates,
                                               requires_grad=True)
        elif initial_coordinates.requires_grad is False:
            initial_coordinates.requires_grad = True

        self.model = model
        if self.model.training:
            warnings.warn('model is in training mode, and certain PyTorch '
                          'layers, such as BatchNorm1d, behave differently '
                          'in training mode in ways that can negatively bias '
                          'simulations. We recommend that you put the model '
                          'into inference mode by calling `model.eval`.')

        self.initial_coordinates = initial_coordinates
        self.embeddings = embeddings
        self.n_sims = self.initial_coordinates.shape[0]
        self.n_beads = self.initial_coordinates.shape[1]
        self.n_dims = self.initial_coordinates.shape[2]

        self.save_forces = save_forces
        self.save_potential = save_potential
        self.length = length
        self.save_interval = save_interval
        self.dt = dt
        self.diffusion = diffusion
        self.beta = beta
        self.verbose = verbose

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed

        self.device = device

        self._simulated = False

    def simulate(self, overwrite=False):
        """Generates independent simulations.

        Parameters
        ----------
        overwrite : Bool (default=False)
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_traj : np.ndarray
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates

        Attributes
        ----------
        simulated_forces : np.ndarray or None
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            If simulated_forces is True, stores the simulation forces
        simulated_potential : np.ndarray or None
            Shape [n_simulations, n_frames, [potential dimensions]]
            If simulated_potential is True, stores the potential calculated
            for each frame in simulation

        """
        if self._simulated and not overwrite:
            raise RuntimeError('Simulation results are already populated. '
                               'To rerun, set overwrite=True.')

        if self.verbose:
            i = 1
            print(
                "Generating {} simulations of length {} at {}-step intervals".format(
                    self.n_sims, self.length, self.save_interval)
            )
        save_size = int(self.length/self.save_interval)

        self.simulated_traj = torch.zeros((save_size, self.n_sims, self.n_beads,
                                           self.n_dims))
        if self.save_forces:
            self.simulated_forces = torch.zeros((save_size, self.n_sims,
                                                 self.n_beads, self.n_dims))
        else:
            self.simulated_forces = None

        self.simulated_potential = None

        # Here if the input is numpy.ndarray, it must be converted to a
        # torch.Tensor with requires_grad=True
        if isinstance(self.initial_coordinates, torch.Tensor):
            x_old = self.initial_coordinates.clone().detach().requires_grad_(True).to(self.device)
        if isinstance(self.initial_coordinates, np.ndarray):
            x_old = torch.tensor(self.initial_coordinates,
                                 requires_grad=True).to(self.device)

        dtau = self.diffusion * self.dt
        for t in range(self.length):
            potential, forces = self.model(x_old, self.embeddings)
            potential = potential.detach()
            forces = forces.detach()
            noise = torch.randn(self.n_sims,
                                self.n_beads,
                                self.n_dims,
                                generator=self.rng).to(self.device)
            x_new = (x_old.detach() + forces*dtau +
                     np.sqrt(2*dtau/self.beta)*noise)

            if t % self.save_interval == 0:
                self.simulated_traj[t//self.save_interval, :, :] = x_new
                if self.save_forces:
                    self.simulated_forces[t//self.save_interval,
                                          :, :] = forces
                if self.save_potential:
                    # The potential will look different for different
                    # network structures, so determine its dimensionality
                    # on the fly
                    if self.simulated_potential is None:
                        assert potential.shape[0] == self.n_sims
                        potential_dims = ([save_size, self.n_sims] +
                                          [potential.shape[j]
                                           for j in range(1,
                                                          len(potential.shape))])
                        self.simulated_potential = torch.zeros(
                            (potential_dims))

                    self.simulated_potential[
                        t//self.save_interval] = potential
            x_old = x_new.clone().detach().requires_grad_(True).to(self.device)

            if self.verbose:
                if t % (self.length/10) == 0 and t > 0:
                    print('{}0% finished'.format(i))
                    i += 1

        if self.verbose:
            print('100% finished.')
        self.simulated_traj = self.swap_axes(
            self.simulated_traj, 0, 1).cpu().numpy()

        if self.save_forces:
            self.simulated_forces = self.swap_axes(self.simulated_forces,
                                                   0, 1).cpu().numpy()

        if self.save_potential:
            self.simulated_potential = self.swap_axes(self.simulated_potential,
                                                      0, 1).cpu().numpy()

        self._simulated = True
        return self.simulated_traj

