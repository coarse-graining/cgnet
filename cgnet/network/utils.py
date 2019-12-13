# Authors: Nick Charron, Brooke Husic, Jiang Wang
# Contributors: Dominik Lemm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from cgnet.feature import GeometryFeature, SchnetFeature, FeatureCombiner


def _schnet_feature_weight_extractor(schnet_feature, return_data=False):
    """Helper function to extract instances of nn.Linear from a SchnetFeature

    Parameters
    ----------
    schnet_feature : SchnetFeature instance
        The SchnetFeature instance from which nn.Linear instances will be
        extracted.
    return_data : bool (default=False)
        If True, the function returns the torch tensor for each weight
        layer rather than the nn.Linear instance.

    Returns
    -------
    linear_list : list of nn.Linear instances or np.arrays,
        The list of nn.Linear layers extracted from the supplied
        SchnetFeature. If 'return_data=True', the function instead returns
        the torch tensors of each nn.Linear instance.

    Notes
    -----
    Add dense layers from each interaction block in the order: intial dense,
    cfconv filter_layers, and output layers, all each within one interaction
    block.
    """

    linear_list = []
    for block in schnet_feature.interaction_blocks:
        linear_list += [layer for layer in block.initial_dense
                        if isinstance(layer, nn.Linear)]
        linear_list += [layer for layer in block.cfconv.filter_generator
                        if isinstance(layer, nn.Linear)]
        linear_list += [layer for layer in block.output_dense
                        if isinstance(layer, nn.Linear)]
    if return_data:
        linear_list = [layer.weight.data for layer in linear_list]
        return linear_list
    else:
        return linear_list


def lipschitz_projection(model, strength=10.0, network_mask=None, schnet_mask=None):
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
    network_mask : list of bool (default=None)
        mask used to exclude certain terminal network layers from lipschitz
        projection. If an element is False, the corresponding weight layer
        is exempt from a lipschitz projection.
    schnet_mask : list of bool (default=None)
        mask used to exclude certain SchnetFeature layers from lipschitz projection.
        If an element is False, the corresponding weight layer is exempt from a
        lipschitz projection. The linear layers of a SchnetFeature InteractionBlock
        have the following arrangement:
            [initial_dense, cfconv_filter #1, cfconv_filter #2, output_dense #1,
             output_dense #2]
        that is, each InteractionBlock contains 5 nn.Linear instances

    Notes
    -----
    L2 Lipshitz regularization is a per-layer regularization that constrains
    the Lipschitz constant of each mapping from one linear layer to the next.
    As formulated by Gouk et. al. (2018), this constraint can be enforced by
    comparing the magnitudes between the weighted dominant singular value of
    the linear layer weight matrix and unity, taking the maximum, and
    normalizing the weight matrix by this result:

        W = W / max( s_dom / lambda, 1.0 )

    for weight matrix W, dominant singular value s_dom, and regularization
    strength lambda. In this form, a strong regularization is achieved for
    lambda -> 0, and a weak regularization is achieved for lambda -> inf.

    References
    ----------
    Gouk, H., Frank, E., Pfahringer, B., & Cree, M. (2018). Regularisation
    of Neural Networks by Enforcing Lipschitz Continuity. arXiv:1804.04368
    [Cs, Stat]. Retrieved from http://arxiv.org/abs/1804.04368
    """

    # Grab all instances of nn.Linear in the model, including those
    # that are part of SchnetFeatures
    # First, we grab the instances of nn.Linear from model.arch
    network_weight_layers = [layer for layer in model.arch
                             if isinstance(layer, nn.Linear)]
    # Next, we grab the nn.Linear instances from the SchnetFeature
    schnet_weight_layers = []
    # if it is part of a FeatureCombiner instance
    if isinstance(model.feature, FeatureCombiner):
        for feature in model.feature.layer_list:
            if isinstance(feature, SchnetFeature):
                schnet_weight_layers += _schnet_feature_weight_extractor(
                    feature)
    # Lastly, we handle the case of SchnetFeatures that are not part of
    # a FeatureCombiner instance
    elif isinstance(model.feature, SchnetFeature):
        schnet_weight_layers += _schnet_feature_weight_extractor(model.feature)
    else:
        pass

    # Next, we assemble a (possibly combined from network and SchnetFeature) mask
    if network_mask is not None:
        if not isinstance(network_mask, list):
            raise ValueError("Lipschitz network mask must be list of booleans")
        if len(network_weight_layers) != len(network_mask):
            raise ValueError("Lipshitz network mask must have the same number "
                             "of elements as the number of nn.Linear "
                             "modules in the model.arch attribute.")
    if network_mask is None:
        network_mask = [True for _ in network_weight_layers]

    if schnet_mask is not None:
        if not isinstance(schnet_mask, list):
            raise ValueError("Lipschitz schnet mask must be list of booleans")
        if len(schnet_weight_layers) != len(schnet_mask):
            raise ValueError("Lipshitz schnet mask must have the same number "
                             "of elements as the number of nn.Linear "
                             "modules in the model SchnetFeature.")
    if schnet_mask is None:
        schnet_mask = [True for _ in schnet_weight_layers]

    full_mask = network_mask + schnet_mask
    full_weight_layers = network_weight_layers + schnet_weight_layers
    for mask_element, layer in zip(full_mask, full_weight_layers):
        if mask_element:
            weight = layer.weight.data
            u, s, v = torch.svd(weight)
            if next(model.parameters()).is_cuda:
                device = weight.device
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]).to(device))
            else:
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]))
            layer.weight.data = weight / (lip_reg)


def dataset_loss(model, loader, optimizer=None,
                 regularization_function=None,
                 verbose_interval=None,
                 print_function=None):
    r"""Compute average loss over arbitrary data loader.
    This can be used during testing, in which `optimizer` and
    `regularization_function` will remain None, or it can be used
    during training, in which an optimizer and (optional)
    regularization_function are provided.

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        model to calculate loss
    loader : torch.utils.data.DataLoader() instance
        loader (with associated dataset)
    optimizer : torch.optim method or None (default=None)
        If not None, the optimizer will be zeroed and stepped for each batch.
    regularization_function : in-place function or None (default=None)
        If not None, the regularization function will be applied after
        stepping the optimizer. It must take only "model" as its input
        and operate in-place.
    verbose_interval : integer or None (default=None)
        If not None, a printout of the batch number and loss will be provided
        at the specified interval (with respect to batch number).
    print_function : python function or None (default=None)
        Print function that takes (batch_number, batch_loss) as its only
        two arguments, to print updates with our default or the style of
        your choice when verbose_interval is not None.

    Returns
    -------
    loss : float
        loss computed over the entire dataset. If the last batch consists of a
        smaller set of left over examples, its contribution to the loss is
        weighted by the ratio of number elements in the MSE matrix to that of
        the normal number of elements associated with the loader's batch size
        before summation to a scalar.

    Example
    -------
    from torch.utils.data import DataLoader

    # assume model is a CGNet object

    # For test data, no optimizer or regularization are needed
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    test_loss = dataset_loss(net, test_data_loader)

    # For training data, an optimizer is needed. Regularization may
    # be used, too
    training_data_loader = DataLoader(training_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Regularization must be in place
    def my_reg_fxn(model, strength=lipschitz_strength):
        lipschitz_projection(model, strength=strength)

    def my_print_fxn(batch_num, batch_loss):
        print("--> Batch #{}, loss = {}".format(batch_num, batch_loss))

    training_loss = dataset_loss(net, training_data_loader,
                                 optimizer = optimizer,
                                 regularization_function = my_reg_fxn,
                                 verbose_interval = 128,
                                 print_function = my_print_fxn)

    Notes
    -----
    This method assumes that if there is a smaller batch, it will be at the
    end: namely, we assume that the size of the first batch is the largest
    batch size.

    """
    if optimizer is None and regularization_function is not None:
        raise RuntimeError(
            "regularization_function is only used when there is an optimizer, "
            "but you have optimizer=None."
        )

    loss = 0
    effective_number_of_batches = 0

    for batch_num, batch_data in enumerate(loader):
        if optimizer is not None:
            optimizer.zero_grad()

        coords, force, embedding_property = batch_data
        if batch_num == 0:
            reference_batch_size = coords.numel()

        batch_weight = coords.numel() / reference_batch_size
        if batch_weight > 1:
            raise ValueError(
                "The first batch was not the largest batch, so you cannot use "
                "dataset loss."
            )

        potential, predicted_force = model.forward(coords,
                                                   embedding_property=embedding_property)

        batch_loss = model.criterion(predicted_force, force)

        if optimizer is not None:
            batch_loss.backward()
            optimizer.step()

            if regularization_function is not None:
                regularization_function(model)

        if verbose_interval is not None:
            if(batch_num + 1) % verbose_interval == 0:
                if print_function is None:
                    print("Batch: {}, Loss: {:.2f}".format(batch_num+1,
                                                           batch_loss))
                else:
                    print_function(batch_num+1, batch_loss)

        loss += batch_loss.cpu().detach().numpy() * batch_weight

        effective_number_of_batches += batch_weight

    loss /= effective_number_of_batches
    return loss


class Simulation():
    """Simulate an artificial trajectory from a CGnet using overdamped Langevin
    dynamics.

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
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out

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
