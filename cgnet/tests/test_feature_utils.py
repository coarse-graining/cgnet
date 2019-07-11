# Author: Dominik Lemm
# Contributors: Brooke Husic, Nick Charron

import numpy as np
import torch

from cgnet.feature.utils import RadialBasisFunction, ShiftedSoftplus

# Random protein
num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)


def test_radial_basis_function():
    # Make sure radial basis functions are consistent with manual calculation

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = torch.randn((num_examples, num_beads, num_beads - 1))
    variance = 1.0

    # Calculate Gaussian expansion using the implemented layer
    rbf = RadialBasisFunction(cutoff=4.0, num_gaussians=6, variance=variance)
    gauss_layer = rbf.forward(distances)

    # Manually calculate expansion
    centers = torch.tensor([0.0000, 0.8000, 1.6000, 2.4000, 3.2000, 4.0000])
    coefficient = -0.5 / variance
    magnitude_squared = (distances.unsqueeze(dim=3) - centers)**2
    gauss_manual = torch.exp(coefficient * magnitude_squared)

    np.testing.assert_allclose(gauss_layer, gauss_manual)


def test_shifted_softplus():
    # Make sure shifted softplus activation is consistent with
    # manual calculation

    # Initialize random feature vector
    feature = torch.randn((num_examples, num_beads), dtype=torch.double)

    ssplus = ShiftedSoftplus()
    manual_output = torch.log(1.0 + torch.exp(feature)) - np.log(2.0)

    np.testing.assert_allclose(manual_output, ssplus(feature))
