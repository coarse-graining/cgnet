# Authors: Nick Charron, Dominik Lemm
# Contributors: Brooke Husic

import torch
from torch import nn as nn


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
        if isinstance(activation, nn.Module):
            seq += [activation]
        else:
            raise TypeError(
                'Activation \"{}\" is not a valid torch.nn.Module'.format(
                    str(activation))
            )
    if dropout:
        seq += [nn.Dropout(dropout)]
    if weight_init == 'xavier':
        torch.nn.init.xavier_uniform_(seq[0].weight)
    if weight_init == 'identity':
        torch.nn.init.eye_(seq[0].weight)
    if weight_init not in ['xavier', 'identity', None]:
        if isinstance(weight_init, int) or isinstance(weight_init, float):
            torch.nn.init.constant_(seq[0].weight, weight_init)
        if callable(weight_init):
            if weight_init_args is None:
                weight_init_args = []
            if weight_init_kwargs is None:
                weight_init_kwargs = []
            weight_init(seq[0].weight, *weight_init_args, **weight_init_kwargs)
        else:
            raise RuntimeError(
                'Unknown weight initialization \"{}\"'.format(str(weight_init))
            )
    return seq

class ShiftedSoftplus(nn.Module):
    r""" Shifted softplus (SSP) activation function

    SSP originates from the softplus function:

        y = \ln\left(1 + e^{-x}\right)

    Schütt et al. (2018) introduced a shifting factor to the function in order
    to ensure that SSP(0) = 0 while having infinite order of continuity:

         y = \ln\left(1 + e^{-x}\right) - \ln(2)

    SSP allows to obtain smooth potential energy surfaces and second derivatives
    that are required for training with forces as well as the calculation of
    vibrational modes (Schütt et al. 2018).

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779

    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input_tensor):
        """ Applies the shifted softplus function element-wise

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (n_examples, *) where `*` means, any number of
            additional dimensions.

        Returns
        -------
        Output: torch.Tensor
            Same shape (n_examples, *) as the input.
        """
        return nn.functional.softplus(input_tensor) - np.log(2.0)


class RadialBasisFunction(nn.Module):
    r"""Radial basis function (RBF) layer

    This layer serves as a distance expansion using radial basis functions with
    the following form:

        e_k (r_j - r_i) = exp(- \gamma (\left \| r_j - r_i \right \| - \mu_k)^2)

    with centers mu_k calculated on a uniform grid between
    zero and the distance cutoff and gamma as a scaling parameter.
    The radial basis function has the effect of decorrelating the
    convolutional filter, which improves the training time.

    Parameters
    ----------
    feat_idx : list(int)
        List of feature idices that serve as a callback to a previous
        featurization layer. For example, these indices could by the
        indices corresponding to all pairwise distances output from a
        ProteinBackboneFeature() layer
    cutoff : float (default=5.0)
        Distance cutoff for the Gaussian function. The cutoff represents the
        center of the last gaussian function in basis.
    num_gaussians : int (default=50)
        Total number of Gaussian functions to calculate. Number will be used to
        create a uniform grid from 0.0 to cutoff. The number of Gaussians will
        also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_neighbors, n_gauss]).
    variance : float (default=1.0)
        The variance (standard deviation squared) of the Gaussian functions.
    """

    def __init__(self, indices, cutoff=5.0, num_gaussians=50, variance=1.0):
        super(RadialBasisFunction, self).__init__()
        self.centers = torch.linspace(0.0, cutoff, num_gaussians)
        self.variance = variance
        self.feat_idx = indices

    def forward(self, distances):
        """Calculate Gaussian expansion

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of shape [n_examples, n_beads, n_neighbors]

        Returns
        -------
        gaussian_exp: torch.Tensor
            Gaussian expansions of shape [n_examples, n_beads, n_neighbors, n_gauss]
        """
        dist_centered_squared = torch.pow(distances.unsqueeze(dim=3) -
                                          self.centers, 2)
        gaussian_exp = torch.exp(-(0.5 / self.variance)
                                 * dist_centered_squared)
        return gaussian_exp
