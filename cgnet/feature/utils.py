# Authors: Nick Charron, Dominik Lemm
# Contributors: Brooke Husic


import numpy as np
import torch
import torch.nn as nn


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
    cutoff : float (default=5.0)
        Distance cutoff for the Gaussian function. The cutoff represents the
        center of the last gaussian function in basis.
    n_gaussians : int (default=50)
        Total number of Gaussian functions to calculate. Number will be used to
        create a uniform grid from 0.0 to cutoff. The number of Gaussians will
        also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_neighbors, n_gauss]).
    variance : float (default=1.0)
        The variance (standard deviation squared) of the Gaussian functions.
    """

    def __init__(self, cutoff=5.0, n_gaussians=50, variance=1.0):
        super(RadialBasisFunction, self).__init__()
        self.register_buffer('centers', torch.linspace(0.0,
                             cutoff, n_gaussians))
        self.variance = variance

    def forward(self, distances):
        """Calculate Gaussian expansion

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of shape [n_examples, n_beads, n_neighbors]

        Returns
        -------
        gaussian_exp: torch.Tensor
            Gaussian expansions of shape [n_examples, n_beads, n_neighbors,
            n_gauss]
        """
        dist_centered_squared = torch.pow(distances.unsqueeze(dim=3) -
                                          self.centers, 2)
        gaussian_exp = torch.exp(-(0.5 / self.variance)
                                 * dist_centered_squared)
        return gaussian_exp


class TelescopingRBF(nn.Module):
    r"""Radial basis function (RBF) layer
    This layer serves as a distance expansion using modulated radial
    basis functions with the following form:

        g_k(r_{ij}) = \phi(r_{ij}, cutoff) *
        exp(- \beta * (\left \exp(-r_{ij}) - \mu_k\right)^2)

    where \phi(r_{ij}, cutoff) is a piecewise polynomial modulation
    function of the following form,

                /
               |    1 - 6*(r_{ij}/cutoff)^5
               |    + 15*(r_{ij}/cutoff)^4      for r_{ij} < cutoff
     \phi = -- |    - 10*(r_{ij}/cutoff)^3
               |
               |    0.0                         for r_{ij} >= cutoff
                \

    the centers mu_k calculated on a uniform grid between
    exp(-r_{ij}) and 1.0, and beta as a scaling parameter defined as:

        \beta = ((2/n_gaussians) * (1 - exp(-cutoff))^-2

    The radial basis function has the effect of decorrelating the
    convolutional filter, which improves the training time.

    Parameters
    ----------
    cutoff : float (default=10.0)
        Distance cutoff (in angstroms) for the modulation. The decay of the
        modulation envelope has positive concavity and smoothly approaches
        zero in the vicinity of the specified cutoff distance. The default
        value of 10 angstroms is taken from Unke & Meuwly (2019). In principle,
        the ideal value should be taken as the largest pairwise distance in the
        system.
    n_gaussians : int (default=64)
        Total number of gaussian functions to calculate. Number will be used to
        create a uniform grid from exp(-cutoff) to 1. The number of gaussians
        will also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_neighbors, n_gauss]). The default value of
        64 gaussians is taken from Unke & Meuwly (2019).
    tolerance : float (default=1e-10)
        When expanding the modulated gaussians, values below the tolerance
        will be set to zero.
    device : torch.device (default=torch.device('cpu'))
        Device upon which tensors are mounted

    Attributes
    ----------
    beta : float
        Gaussian decay parameter, defined as:
            \beta = ((2/n_gaussians) * (1 - exp(-cutoff))^-2

    Notes
    -----
    These basis functions were originally introduced as part of the PhysNet
    architecture (Unke & Meuwly, 2019). Though the basis function centers are
    scattered uniformly, the modulation function has the effect of broadening
    those functions closer to the specified cutoff. The overall result is a set
    of basis functions which have high resolution at small distances which
    smoothly morphs to basis functions with lower resolution at larger
    distances.

    References
    ----------
    Unke, O. T., & Meuwly, M. (2019). PhysNet: A Neural Network for Predicting
        Energies, Forces, Dipole Moments and Partial Charges. Journal of
        Chemical Theory and Computation, 15(6), 3678–3693.
        https://doi.org/10.1021/acs.jctc.9b00181

    """

    def __init__(self, cutoff=10.0, n_gaussians=64, tolerance=1e-10,
                 device=torch.device('cpu')):
        super(TelescopingRBF, self).__init__()
        self.tolerance = tolerance
        self.device = device
        self.register_buffer('centers', torch.linspace(np.exp(-cutoff), 1,
                             n_gaussians))
        self.cutoff = cutoff
        self.beta = np.power(((2/n_gaussians)*(1-np.exp(-self.cutoff))), -2)

    def modulation(self, distances):
        """PhysNet cutoff modulation function

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of shape [n_examples, n_beads, n_neighbors]

        Returns
        -------
        mod : torch.Tensor
            The modulation envelope of the radial basis functions. Shape
            [n_examples, n_beads, n_neighbors]

        """
        zeros = torch.zeros_like(distances).to(self.device)
        modulation_envelope = torch.where(distances < self.cutoff,
                                          1 - 6 *
                                          torch.pow((distances/self.cutoff), 5)
                                          + 15 *
                                          torch.pow((distances/self.cutoff), 4)
                                          - 10 *
                                          torch.pow(
                                              (distances/self.cutoff), 3),
                                          zeros)
        return modulation_envelope

    def forward(self, distances):
        """Calculate modulated gaussian expansion

        Parameters
        ----------
        distances : torch.Tensor
            Interatomic distances of shape [n_examples, n_beads, n_neighbors]

        Returns
        -------
        expansions : torch.Tensor
            Modulated gaussian expansions of shape
            [n_examples, n_beads, n_neighbors, n_gauss]

        Notes
        -----
        The gaussian portion of the basis function is a function of
        exp(-r_{ij}), not r_{ij}

        """
        dist_centered_squared = torch.pow(torch.exp(-distances.unsqueeze(dim=3))
                                          - self.centers, 2)
        gaussian_exp = torch.exp(-self.beta
                                 * dist_centered_squared)
        modulation_envelope = self.modulation(distances).unsqueeze(dim=3)

        expansions = modulation_envelope * gaussian_exp

        # In practice, this gives really tiny numbers. For numbers below the
        # tolerance, we just set them to zero.
        expansions = torch.where(torch.abs(expansions) > self.tolerance,
                                 expansions,
                                 torch.zeros_like(expansions))

        return expansions


def LinearLayer(
        d_in,
        d_out,
        bias=True,
        activation=None,
        dropout=0,
        weight_init='xavier',
        weight_init_args=None,
        weight_init_kwargs=None):
    r"""Linear layer function

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
    MyLayer = LinearLayer(5, 10, bias=True, activation=nn.Softplus(beta=2),
                          weight_init=nn.init.kaiming_uniform_,
                          weight_init_kwargs={"a":0, "mode":"fan_out",
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
                'Activation {} is not a valid torch.nn.Module'.format(
                    str(activation))
            )
    if dropout:
        seq += [nn.Dropout(dropout)]

    with torch.no_grad():
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
                weight_init(seq[0].weight, *weight_init_args,
                            **weight_init_kwargs)
            else:
                raise RuntimeError(
                    'Unknown weight initialization \"{}\"'.format(
                        str(weight_init))
                )
    return seq
