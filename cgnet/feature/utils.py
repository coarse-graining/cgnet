import numpy as np
import torch
import torch.nn as nn


class ShiftedSoftplus(nn.Module):
    """
    Shifted soft-plus activation function with the form:

        y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Guarantees smooth derivates and conserves zero-activations.

    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input_tensor):
        """ Applies the shifted soft-plus function element-wise

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (n_examples, *) where `*` means, any number of
            additional dimensions

        Returns
        -------
        Output: torch.Tensor
            Same shape (n_examples, *) as the input.
        """
        return nn.functional.softplus(input_tensor) - np.log(2.0)


class RadialBasisFunction(nn.Module):
    """Radial basis function layer

    This layer serves as a distance expansion using radial basis functions with
    the following form:

        e_k (r_j - r_i) = exp(- \gamma (\left \| r_j - r_i \right \| - \mu_k)^2)

    with mu_k being the centers chosen on a uniform grid and gamma as  a scaling
    parameter. The radial basis function has the effect of decorellating the
    convolutional filter, which improves the training time.

    Parameters
    ----------
    cutoff : float (default 5.0)
        Distance cutoff for the Gaussian function. The cutoff represents the
        center of the last gaussian function in basis.
    num_gaussians : int (default 50)
        Total number of Gaussian functions to calculate. Number will be used to
        create a uniform grid from 0.0 to cutoff. The number of Gaussians will
        also decide the output size of the RBF layer output
        ([n_examples, n_beads, n_nbh, n_gauss]).
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
            Interatomic distances of shape [n_examples, n_beads, n_nbh]

        Returns
        -------
        gaussian_exp: torch.Tensor
            Gaussian expansions of shape [n_examples, n_beads, n_nbh, n_gauss]
        """
        dist_centered_squared = torch.pow(distances.unsqueeze(dim=3) -
                                      self.centers, 2)
        gaussian_exp = torch.exp(-(0.5 / self.variance) * dist_centered_squared)
        return gaussian_exp
