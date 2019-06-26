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
        #print(coords.size()[0] / loader.batch_size)
        # print(coords.size()[0])
        loss += model.criterion(pred_force,
                                force) * (coords.numel() / ref_numel)
        num_batch += (coords.numel() / ref_numel)
    loss /= num_batch
    return loss.data.item()


class Simulation():
    """Simulate an artificial trajectory from a CGnet.

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        model to calculate loss
    initial_coordinates : np.array
            Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
            Each entry in the first dimension represents the first frame of an
            independent simulation.
    length : int (default=100)
            The length of the simulation in simulation timesteps
    save_interval : int (default=10)
            The interval at which simulation timesteps should be saved
    dt : float (default=5e-4)
            TODO
    beta : float (default=0.01)
            TODO
    verbose : bool (default=False)
            Whether to print simulation progress information

    Notes
    -----
    Long simulation lengths may take a significant amount of time.
    """

    def __init__(self, model, initial_coordinates,
                 length=100, save_interval=10, dt=5e-4,
                 beta=0.01, verbose=False):
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

        self.length = length
        self.save_interval = save_interval
        self.dt = dt
        self.beta = beta
        self.verbose = verbose

    def simulate(self):
        """Generates independent simulations.

        Returns
        -------
        simulated_traj : np.array
            Dimensions [n_simulations, n_frames, n_atoms, n_dimensions]
        """
        if self.verbose:
            i = 1
            print(
        "Generating {} simulations of length {} at {}-step intervals".format(
                    self.n_sims, self.length, self.save_interval)
            )
        self.simulated_traj = np.zeros((int(self.length/self.save_interval),
                                        self.n_sims, self.n_beads, self.n_dims))
        x_old = self.initial_coordinates
        for t in range(self.length):
            _, forces = self.model(x_old)
            noise = torch.tensor(np.random.randn(self.n_sims,
                                                 self.n_beads,
                                                 self.n_dims)).float()
            x_new = x_old + forces*self.dt + np.sqrt(2*self.dt/self.beta)*noise
            if t % self.save_interval == 0:
                # print(forces)
                self.simulated_traj[t//self.save_interval,
                                    :, :] = x_new.detach().numpy()
            x_old = x_new
            if t % (self.length/10) == 0:
                print('{}0% finished'.format(i))
                i += 1

        self.simulated_traj = np.swapaxes(self.simulated_traj, 0, 1)
        return self.simulated_traj
