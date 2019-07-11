# Author: Dominik Lemm
# Contributors: Brooke Husic, Nick Charron

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
    num_gaussians : int (default=50)
        Total number of Gaussian functions to calculate. Number will be used to
        create a uniform grid from 0.0 to cutoff. The number of Gaussians will
        also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_neighbors, n_gauss]).
    variance : float (default=1.0)
        The variance (standard deviation squared) of the Gaussian functions.
    """

    def __init__(self, cutoff=5.0, num_gaussians=50, variance=1.0):
        super(RadialBasisFunction, self).__init__()
        self.centers = torch.linspace(0.0, cutoff, num_gaussians)
        self.variance = torch.FloatTensor(
            variance * torch.ones_like(self.centers))

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
