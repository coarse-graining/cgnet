# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang

import torch
import torch.nn as nn
import numpy as np


class ForceLoss(torch.nn.Module):
    """Loss function for force matching scheme."""

    def __init__(self):
        super(ForceLoss, self).__init__()

    def forward(self, force, labels):
        """Returns force matching loss averaged over all examples.

        Parameters
        ----------
        force : torch.Tensor (grad enabled)
            forces calculated from the CGnet energy via autograd.
            Size [n_examples, n_degrees_freedom].
        labels : torch.Tensor
            forces to compute the loss against. Size [n_examples,
                                                      n_degrees_of_freedom].

        Returns
        -------
        loss : torch.Variable
            example-averaged Frobenius loss from force matching. Size [1, 1].

        """
        loss = ((force - labels)**2).mean()
        return loss


def LinearLayer(
        d_in,
        d_out,
        bias=True,
        activation=None,
        dropout=0,
        weight_init='xavier',
        weight_init_args=None,
        weight_init_kwargs=None):
    """Linear layer function

    Parameters
    ----------
    d_in : int
        input dimension
    d_out : int
        output dimension
    bias : bool (default=True)
        specifies whether or not to add a bias node
    activation : torch.nn.Module() (default=None)
        activation function for the layer
    dropout : float (default=0)
        if > 0, a dropout layer with the specified dropout frequency is
        added after the activation.
    weight_init : str, float, or nn.init function (default=\'xavier\')
        specifies the initialization of the layer weights. For non-option
        initializations (eg, xavier initialization), a string may be used
        for simplicity. If a float or int is passed, a constant initialization
        is used. For more complicated initializations, a torch.nn.init function
        object can be passed in.
    weight_init_args : list or tuple (default=None)
        arguments (excluding the layer.weight argument) for a torch.nn.init
        function.
    weight_init_kwargs : dict (default=None)
        keyword arguements for a torch.nn.init function

    Returns
    -------
    seq : list of torch.nn.Module() instances
        the full linear layer, including activation and optional dropout.

    Example
    -------
    MyLayer = LinearLayer(5,10,bias=True,activation=nn.Softplus(beta=2),
                               weight_init=nn.init.kaiming_uniform_,
                               weight_init_kwargs={"a":0,"mode":"fan_out",
                               "nonlinearity":"leaky_relu"})

    Produces a linear layer with input dimension 5, output dimension 10, bias
    inclusive, followed by a beta=2 softplus activation, with the layer weights
    intialized according to kaiming uniform procedure with preservation of weight
    variance magnitudes during backpropagation.

    """

    seq = [nn.Linear(d_in, d_out, bias=bias)]
    if activation:
        seq += [activation]
    if dropout:
        seq += [nn.Dropout(dropout)]
    if weight_init == 'xavier':
        torch.nn.init.xavier_uniform_(seq[0].weight)
    if weight_init == 'identity':
        torch.nn.init.eye_(seq[0].weight)
    if isinstance(weight_init, int) or isinstance(weight_init, float):
        torch.nn.init.constant_(seq[0].weight, weight_init)
    if callable(weight_init):
        if weight_init_args is None:
            weight_init_args = []
        if weight_init_kwargs is None:
            weight_inti_kwargs = []
        weight_init(seq[0].weight, *weight_init_args, **weight_init_kwargs)
    return seq

class RepulsionLayer(nn.Module):
    """Layer for calculating pairwise repulsion energy prior
    Parameters
    ----------
    feat_data: list
        list of distance tuples for which to calculate repulsion interactions
    excluded_volume: float (default=5.5)
        Excluded volume parameter.
    exponent: float (default=6.0)
        Exponent of repulsion interaction. By convention, this value is
        taken to be positive.
    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys.
    feature_type: str
        features type from which to select coordinates.
    """

    def __init__(self, feat_data, excluded_volume=5.5, exponent=6.0,
                 descriptions=None, feature_type=None):
        super(RepulsionLayer, self).__init__()
        self.excluded_volume = excluded_volume
        self.exponent = exponent
        if descriptions and not feature_type:
            raise RuntimeError('Must declare feature_type if using \
                                descriptions')
        if descriptions and feature_type:
            self.feature_type = feature_type
            self.features = feat_data
            self.feat_idx = []
            # get number of each feature to determine starting idx
            nums = [len(descriptions['Distances']), len(descriptions['Angles']),
                    len(descriptions['Dihedral_cosines']),
                    len(descriptions['Dihedral_sines'])]
            descs = ['Distances', 'Angles', 'Dihedral_cosines', 'Dihedral_sines']
            start_idx = 0
            for num, desc in zip(nums, descs):
                if self.feature_type == desc:
                    break
                else:
                    start_idx += num
            for feat in self.features:
                self.feat_idx.append(start_idx +
                                 descriptions[self.feature_type].index(feat))

    def forward(self, in_feat):
        """Calculates repulsion interaction contributions to energy
        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as pairwise distances, of size (n,k), for
            n examples and k features.
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum((self.excluded_volume/in_feat) ** self.exponent,
                            1).reshape(n, 1) / 2
        return energy

class HarmonicLayer(nn.Module):
    """Layer for calculating bond/angle harmonic energy prior

    Parameters
    ----------
    feat_data: dict
        dictionary of means and bond constants. Keys are tuples that provide the
        descriptions of each contributing feature. THe values of each key are in
        turn dictionarys that have the following keys. The \'mean\' key is mapped
        to the numerical mean of the feature over the trajectory. The \'std\' key
        is mapped to the numerical standard deviation of the feature over the
        trajectory. The \'k\' is mapped to the harmonic constant derived from the
        feature.
    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys.
    feature_type: str
        features type from which to select coordinates.

    """

    def __init__(self, feat_data, descriptions=None, feature_type=None):
        super(HarmonicLayer, self).__init__()
        if descriptions and not feature_type:
            raise RuntimeError('Must declare feature_type is using \
                                descriptions')
        if descriptions and feature_type:
            self.feature_type = feature_type
            self.features = []
            self.feat_idx = []
            self.harmonic_parameters = torch.tensor([])
            # get number of each feature to determine starting idx
            nums = [len(descriptions['Distances']), len(descriptions['Angles']),
                    len(descriptions['Dihedral_cosines']),
                    len(descriptions['Dihedral_sines'])]
            descs = ['Distances', 'Angles', 'Dihedral_cosines', 'Dihedral_sines']
            start_idx = 0
            for num, desc in zip(nums, descs):
                if self.feature_type == desc:
                    break
                else:
                    start_idx += num
            for key, params in feat_data.items():
                self.features.append(key)
                self.feat_idx.append(start_idx +
                                 descriptions[self.feature_type].index(key))
                self.harmonic_parameters = torch.cat((self.harmonic_parameters,
                     torch.tensor([[params['k']], [params['mean']]])), dim=1)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum(
            self.harmonic_parameters[0, :] * (in_feat -
                                              self.harmonic_parameters[1, :]) ** 2, 1).reshape(n, 1) / 2
        return energy


class CGnet(nn.Module):
    """CGnet neural network class

    Parameters
    ----------
    arch : list of nn.Module() instances
        underlying sequential network architecture.
    criterion : nn.Module() instances
        loss function to be used for network.
    feature : nn.Module() insstance
        feature layer to transform cartesian coordinates into roto-
        translationalyl invariant features.
    priors : list of nn.Module() instances (default=None)
        list of prior layers that provide energy contributions external to
        the hidden architecture of the CGnet.

    Notes
    -----
    CGnets are a class of feedforward neural networks introduced by Jiang et.
    al. (2019) which are used to predict coarse-grained molecular force fields
    from Cartesain coordinate data. They are characterized by an autograd layer
    with respect to input coordinates implemented before the loss function,
    which directs the network to learn a representation of the coarse-grained
    potential of mean force (PMF) associated with a conservative coarse-grained
    force feild via a gradient operation as prescribed by classical mechanics.
    CGnets may also contain featurization layers, which transform Cartesian
    inputs into roto-translationally invariant features, thereby yeilding a PMF
    that respects these invarainces. CGnets may additionally be supplied with
    external prior functions, which are useful for regularizing network behavior
    in sparsely sampled, unphysical regions of molecular configuration space.

    Examples
    --------
    From Jiang et. al. (2019), the optimal architecture for a 5-bead coarse
    grain model of alanine dipeptide, featurized into bonds, angles, pairwaise
    distances, and backbone torsions, was found to be:

    CGnet(
      (input): in_features=30
      (arch): Sequential(
        (0): ProteinBackboneFeature(in_features=30, out_features=17)
        (1): Linear(in_features=17, out_features=160, bias=True)
        (2): Tanh()
        (3): Linear(in_features=160, out_features=160, bias=True)
        (4): Tanh()
        (5): Linear(in_features=160, out_features=160, bias=True)
        (6): Tanh()
        (7): Linear(in_features=160, out_features=160, bias=True)
        (8): Tanh()
        (9): Linear(in_features=160, out_features=160, bias=True)
        (10): Tanh()
        (11): Linear(in_features=160, out_features=1, bias=True)
        (12): torch.sum((11) + BondPotential(bonds, angles))
        (13): torch.autograd.grad(-(12), input, create_graph=True,
                                  retain_graph=True)
      )
    (criterion): ForceLoss()
    )

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E.,
        de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913

    """

    def __init__(self, arch, criterion, feature=None, priors=None):
        super(CGnet, self).__init__()

        self.arch = nn.Sequential(*arch)
        if priors:
            self.priors = nn.Sequential(*priors)
        else:
            self.priors = None
        self.criterion = criterion
        self.feature = feature

    def forward(self, coord):
        """Forward pass through the network ending with autograd layer.

        Parameters
        ----------
        coord : torch.Tensor (grad enabled)
            input trajectory/data of size [n_examples, n_degrees_of_freedom].

        Returns
        -------
        energy : torch.Tensor
            scalar potential energy of size [n_examples, 1]. If priors are
            supplied to the CGnet, then this energy is the sum of network
            and prior energies.
        force  : torch.Tensor
            vector forces of size [n_examples, n_degrees_of_freedom].
        """
        feat = coord
        if self.feature:
            feat = self.feature(feat)

        # forward pass through the hidden architecture of the CGnet
        energy = self.arch(feat)
        # addition of external priors to form total energy
        if self.priors:
            for prior in self.priors:
                energy += prior(feat[:, prior.feat_idx])

        # Perform autograd to learn potential of conservative force field
        force = torch.autograd.grad(-torch.sum(energy),
                                    coord,
                                    create_graph=True,
                                    retain_graph=True)
        return energy, force[0]

    def predict(self, coord, force_labels):
        """Prediction over test/validation batch.

        Parameters
        ----------
        coord: torch.Tensor (grad enabled)
            input trajectory/data of size [n_examples, n_degrees_of_freedom]
        force_labels: torch.Tensor
            force labels of size [n_examples, n_degrees_of_freedom]

        Returns
        -------
        loss.data : torch.Tensor
            loss over prediction inputs.

        """

        self.eval()  # set model to eval mode
        energy, force = self.forward(coord)
        loss = self.criterion.forward(force, force_labels)
        self.train()  # set model to train mode
        return loss.data
