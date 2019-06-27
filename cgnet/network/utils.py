# Authors: Nick Charron, Brooke Husic, Jiang Wang

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


def lipschitz_projection(model, strength=10.0):
    """Performs L2 Lipschitz Projection via spectral normalization

    Parameters
    ----------
    model : cgnet.network.CGnet() instance
        model to perform Lipschitz projection upon
    strength : float (default=10.0)
        Strength of L2 lipschitz projection via spectral normalization.
        The magntitude of {dominant weight matrix eigenvalue / strength}
        is compared to unity, and the weight matrix is rescaled by the max
        of this comparison

    References
    ----------
    Gouk, H., Frank, E., Pfahringer, B., & Cree, M. (2018). Regularisation
    of Neural Networks by Enforcing Lipschitz Continuity. arXiv:1804.04368
    [Cs, Stat]. Retrieved from http://arxiv.org/abs/1804.04368
    """
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            u, s, v = torch.svd(weight)
            if next(model.parameters()).is_cuda:
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]).cuda())
            else:
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]))
            layer.weight.data = weight / (lip_reg)


def dataset_loss(model, loader):
    """Compute average loss over arbitrary loader/dataset

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        model to calculate loss
    loader : torch.utils.data.DataLoader() instance
        loader (with associated dataset)

    Returns
    -------
    loss : float
        loss computed over the entire dataset. If the last batch consists of a
        smaller set of left over examples, its contribution to the loss is
        weighted by the ratio of number elements in the MSE matrix to that of
        the normal number of elements assocatied with the loader's batch size
        before summation to a scalar.

    Example
    -------
    test_set = MoleculeDataset(coords[test_indices], forces[test_indices])
    test_sampler = torch.utils.data.RandomSubSetSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_sampler,
                                              batch_size=512)
    test_error = dataset_loss(MyModel, test_loader)

    """
    loss = 0
    num_batch = 0
    ref_numel = 0
    for num, batch in enumerate(loader):
        coords, force = batch
        if num == 0:
            ref_numel = coords.numel()
        potential, pred_force = model.forward(coords)
        loss += model.criterion(pred_force,
                                force) * (coords.numel() / ref_numel)
        num_batch += (coords.numel() / ref_numel)
    loss /= num_batch
    return loss.data.item()


class Simulation():
    """Simulate an artificial trajectory from a CGnet using overdamped Langevin
    dynamics.

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        model to calculate loss
    initial_coordinates : np.ndarray
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Units are determined
        by the frame striding of the original training data simulation
    diffusion : float (default=1.0)
        The constant diffusion parameter for overdamped Langevin dynamics. By
        default, the diffusion is set to unity and is absorbed into the dt
        argument. However, users may specify separate diffusion and dt
        parameters in the case that they have some estimate of the CG diffusion
    beta : float (default=0.01)
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
    verbose : bool (default=False)
        Whether to print simulation progress information
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed

    Notes
    -----
    A system evolves under Langevin dyanmics using the following, stochastic
    differential equation:

        dX_t = - grad( U( X_t ) ) * a * dt + sqrt( 2 * a * dt / beta ) * dW_t

    for coordinates X_t at time t, potential energy U, diffusion a,
    thermodynamic inverse temperature beta, time step dt, and stochastic Weiner
    process W. The choice of Langevin dynamics is made because CG systems
    possess no explicit solvent, and so Brownian-like collisions must be
    modeled indirectly using a stochastic term.

    Long simulation lengths may take a significant amount of time.
    """

    def __init__(self, model, initial_coordinates, save_forces=False,
                 save_potential=False, length=100, save_interval=10, dt=5e-4,
                 diffusion=1.0, beta=1.0, verbose=False, random_seed=None):
        if length % save_interval != 0:
            raise ValueError(
                'The save_interval must be a factor of the simulation length'
                )

        self.model = model

        if len(initial_coordinates.shape) != 3:
            raise ValueError(
                'initial_coordinates shape must be [frames, atoms, dimensions]'
            )

        if type(initial_coordinates) is not torch.Tensor:
            initial_coordinates = torch.tensor(initial_coordinates,
                                               requires_grad=True)
        elif initial_coordinates.requires_grad is False:
            initial_coordinates.requires_grad = True

        self.initial_coordinates = initial_coordinates
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
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(random_seed)
        self.random_seed = random_seed

    def simulate(self):
        """Generates independent simulations.

        Returns
        -------
        simulated_traj : np.ndarray
            Dimensions [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates

        Attributes
        ----------
        simulated_forces : np.ndarray or None
            Dimensions [n_simulations, n_frames, n_atoms, n_dimensions]
            If simulated_forces is True, stores the simulation forces
        simulated_potential : np.ndarray or None
            Dimensions [n_simulations, n_frames, [potential dimensions]]
            If simulated_potential is True, stores the potential calculated
            for each frame in simulation 

        """
        if self.verbose:
            i = 1
            print(
                "Generating {} simulations of length {} at {}-step intervals".format(
                    self.n_sims, self.length, self.save_interval)
            )
        save_size = int(self.length/self.save_interval)

        self.simulated_traj = np.zeros((save_size, self.n_sims, self.n_beads,
                                        self.n_dims))
        if self.save_forces:
            self.simulated_forces = np.zeros((save_size, self.n_sims,
                                              self.n_beads, self.n_dims))
        else:
            self.simulated_forces = None

        self.simulated_potential = None

        x_old = self.initial_coordinates
        dtau = self.diffusion * self.dt

        for t in range(self.length):
            potential, forces = self.model(x_old)
            noise = torch.tensor(self.rng.randn(self.n_sims,
                                                 self.n_beads,
                                                 self.n_dims)).float()
            x_new = x_old + forces*dtau + np.sqrt(2*dtau/self.beta)*noise
            if t % self.save_interval == 0:
                self.simulated_traj[t//self.save_interval,
                                    :, :] = x_new.detach().numpy()
                if self.save_forces:
                    self.simulated_forces[t//self.save_interval,
                                          :, :] = forces.detach().numpy()
                if self.save_potential:
                    # The potential will look different for different
                    # network structures, so determine its dimensionality
                    # on the fly
                    if self.simulated_potential is None:
                        assert potential.shape[0] == self.n_sims
                        potential_dims = ([save_size, self.n_sims] +
                                          [potential.shape[j]
                                           for j in range(1, len(potential.shape))])
                        self.simulated_potential = np.zeros((potential_dims))

                    self.simulated_potential[
                        t//self.save_interval] = potential.detach().numpy()
            x_old = x_new

            if self.verbose:
                if t % (self.length/10) == 0:
                    print('{}0% finished'.format(i))
                    i += 1

        self.simulated_traj = np.swapaxes(self.simulated_traj, 0, 1)

        if self.save_forces:
            self.simulated_forces = np.swapaxes(self.simulated_forces, 0, 1)

        if self.save_potential:
            self.simulated_potential = np.swapaxes(self.simulated_potential, 0, 1)

        return self.simulated_traj
