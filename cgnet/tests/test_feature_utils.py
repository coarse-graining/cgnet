# Author: Dominik Lemm
# Contributors: Brooke Husic, Nick Charron

import numpy as np
import torch

from cgnet.feature.utils import RadialBasisFunction, ShiftedSoftplus
from cgnet.feature.statistics import GeometryStatistics
from cgnet.feature.feature import GeometryFeature


# Define sizes for a pseudo-dataset
frames = np.random.randint(10, 30)
beads = np.random.randint(5, 10)



def test_radial_basis_function():
    # Make sure radial basis functions are consistent with manual calculation

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = torch.randn((frames, beads, beads - 1))
    # Define random parameters for the RBF
    variance = np.random.random() + 1
    n_gaussians = np.random.randint(5, 10)
    cutoff = np.random.uniform(1.0, 5.0)

    # Calculate Gaussian expansion using the implemented layer
    rbf = RadialBasisFunction(cutoff=cutoff, n_gaussians=n_gaussians,
                              variance=variance)
    gauss_layer = rbf.forward(distances)

    # Manually calculate expansion with numpy
    # according to the following formula:
    # e_k (r_j - r_i) = exp(- \gamma (\left \| r_j - r_i \right \| - \mu_k)^2)
    # with centers mu_k calculated on a uniform grid between
    # zero and the distance cutoff and gamma as a scaling parameter.
    centers = np.linspace(0.0, cutoff, n_gaussians)
    gamma = -0.5 / variance
    distances = np.expand_dims(distances, axis=3)
    magnitude_squared = (distances - centers)**2
    gauss_manual = np.exp(gamma * magnitude_squared)

    # Shapes and values need to be the same
    np.testing.assert_equal(centers.shape, rbf.centers.shape)
    np.testing.assert_allclose(gauss_layer.numpy(), gauss_manual, rtol=1e-5)


def test_shifted_softplus():
    # Make sure shifted softplus activation is consistent with
    # manual calculation

    # Initialize random feature vector
    feature = torch.randn((frames, beads), dtype=torch.double)

    ssplus = ShiftedSoftplus()
    # Shifted softplus has the following form:
    # y = \ln\left(1 + e^{-x}\right) - \ln(2)
    manual_output = np.log(1.0 + np.exp(feature.numpy())) - np.log(2.0)

    np.testing.assert_allclose(manual_output, ssplus(feature).numpy())

