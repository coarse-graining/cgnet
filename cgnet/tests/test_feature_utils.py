# Author: Dominik Lemm
# Contributors: Brooke Husic, Nick Charron

import numpy as np
import torch
from nose.tools import raises

from cgnet.feature.utils import (GaussianRBF, PolynomialCutoffRBF,
                                 ShiftedSoftplus, _AbstractRBFLayer)
from cgnet.feature.statistics import GeometryStatistics
from cgnet.feature.feature import GeometryFeature, Geometry


# Define sizes for a pseudo-dataset
frames = np.random.randint(10, 30)
beads = np.random.randint(5, 10)
g = Geometry(method='torch')


@raises(NotImplementedError)
def test_radial_basis_function_len():
    # Make sure that a NotImplementedError is raised if an RBF layer
    # does not have a __len__() method

    # Here, we use the _AbstractRBFLayer base class as our RBF
    abstract_RBF = _AbstractRBFLayer()

    # Next, we check to see if the NotImplementedError is raised
    # This is done using the decorator above, because we cannot
    # use nose.tools.assert_raises directly on special methods
    len(abstract_RBF)


def test_radial_basis_function():
    # Make sure radial basis functions are consistent with manual calculation

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = torch.randn((frames, beads, beads - 1), dtype=torch.float64)
    # Define random parameters for the RBF
    variance = np.random.random() + 1
    n_gaussians = np.random.randint(5, 10)
    high_cutoff = np.random.uniform(5.0, 10.0)
    low_cutoff = np.random.uniform(0.0, 4.0)

    # Calculate Gaussian expansion using the implemented layer
    rbf = GaussianRBF(high_cutoff=high_cutoff, low_cutoff=low_cutoff,
                      n_gaussians=n_gaussians, variance=variance)
    gauss_layer = rbf.forward(distances)

    # Manually calculate expansion with numpy
    # according to the following formula:
    # e_k (r_j - r_i) = exp(- \gamma (\left \| r_j - r_i \right \| - \mu_k)^2)
    # with centers mu_k calculated on a uniform grid between
    # zero and the distance cutoff and gamma as a scaling parameter.
    centers = np.linspace(low_cutoff, high_cutoff,
                          n_gaussians).astype(np.float64)
    gamma = -0.5 / variance
    distances = np.expand_dims(distances, axis=3)
    magnitude_squared = (distances - centers)**2
    gauss_manual = np.exp(gamma * magnitude_squared)

    # Shapes and values need to be the same
    np.testing.assert_equal(centers.shape, rbf.centers.shape)
    np.testing.assert_allclose(gauss_layer.numpy(), gauss_manual, rtol=1e-5)


def test_radial_basis_function_distance_masking():
    # Makes sure that if a distance mask is used, the corresponding
    # expanded distances returned by GaussianRBF are zero

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = torch.randn((frames, beads, beads - 1), dtype=torch.float64)
    # Define random parameters for the RBF
    variance = np.random.random() + 1
    high_cutoff = np.random.uniform(5.0, 10.0)
    low_cutoff = np.random.uniform(0.0, 4.0)
    n_gaussians = np.random.randint(5, 10)
    neighbor_cutoff = np.abs(np.random.rand())
    neighbors, neighbor_mask = g.get_neighbors(distances,
                                               cutoff=neighbor_cutoff)

    # Calculate Gaussian expansion using the implemented layer
    rbf = GaussianRBF(high_cutoff=high_cutoff, low_cutoff=low_cutoff,
                      n_gaussians=n_gaussians, variance=variance)
    gauss_layer = rbf.forward(distances, distance_mask=neighbor_mask)

    # Lastly, we check to see that the application of the mask is correct
    # against a manual calculation and masking
    centers = np.linspace(low_cutoff, high_cutoff, n_gaussians)
    gamma = -0.5 / variance
    distances = np.expand_dims(distances, axis=3)
    magnitude_squared = (distances - centers)**2
    gauss_manual = torch.tensor(np.exp(gamma * magnitude_squared))
    gauss_manual = gauss_manual * neighbor_mask[:, :, :, None].double()

    np.testing.assert_array_almost_equal(gauss_layer.numpy(),
                                         gauss_manual.numpy())


def test_radial_basis_function_normalize():
    # Tests to make sure that the output of GaussianRBF is properly
    # normalized if 'normalize_output' is specified as True

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = torch.randn((frames, beads, beads - 1), dtype=torch.float64)
    # Define random parameters for the RBF
    variance = np.random.random() + 1
    n_gaussians = np.random.randint(5, 10)
    high_cutoff = np.random.uniform(5.0, 10.0)
    low_cutoff = np.random.uniform(0.0, 4.0)

    # Calculate Gaussian expansion using the implemented layer
    rbf = GaussianRBF(high_cutoff=high_cutoff, low_cutoff=low_cutoff,
                      n_gaussians=n_gaussians, variance=variance,
                      normalize_output=True)
    gauss_layer = rbf.forward(distances)

    # Manually calculate expansion with numpy
    # according to the following formula:
    # e_k (r_j - r_i) = exp(- \gamma (\left \| r_j - r_i \right \| - \mu_k)^2)
    # with centers mu_k calculated on a uniform grid between
    # zero and the distance cutoff and gamma as a scaling parameter.
    centers = np.linspace(low_cutoff, high_cutoff,
                          n_gaussians).astype(np.float64)
    gamma = -0.5 / variance
    distances = np.expand_dims(distances, axis=3)
    magnitude_squared = (distances - centers)**2
    gauss_manual = np.exp(gamma * magnitude_squared)

    # manual output normalization
    gauss_manual = gauss_manual / np.sum(gauss_manual, axis=3)[:, :, :, None]

    # Shapes and values need to be the same
    np.testing.assert_equal(centers.shape, rbf.centers.shape)
    np.testing.assert_allclose(gauss_layer.numpy(), gauss_manual, rtol=1e-5)


def test_polynomial_cutoff_rbf():
    # Make sure the polynomial_cutoff radial basis functions are consistent with
    # manual calculations

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = np.random.randn(frames, beads, beads - 1).astype(np.float64)
    # Define random parameters for the polynomial_cutoff RBF
    n_gaussians = np.random.randint(5, 10)
    high_cutoff = np.random.uniform(5.0, 10.0)
    low_cutoff = np.random.uniform(0.0, 4.0)
    alpha = np.random.uniform(0.1, 1.0)

    # Calculate Gaussian expansion using the implemented layer
    polynomial_cutoff_rbf = PolynomialCutoffRBF(high_cutoff=high_cutoff,
                                                low_cutoff=low_cutoff,
                                                n_gaussians=n_gaussians,
                                                alpha=alpha,
                                                tolerance=1e-8)
    polynomial_cutoff_rbf_layer = polynomial_cutoff_rbf.forward(
        torch.tensor(distances))

    # Manually calculate expansion with numpy
    # First, we compute the centers and the scaling factors
    centers = np.linspace(np.exp(-high_cutoff), np.exp(-low_cutoff),
                          n_gaussians).astype(np.float64)
    beta = np.power(((2/n_gaussians) * (1-np.exp(-high_cutoff))), -2)

    # Next, we compute the gaussian portion
    exp_distances = np.exp(-alpha * np.expand_dims(distances, axis=3))
    magnitude_squared = np.power(exp_distances - centers, 2)
    gauss_manual = np.exp(-beta * magnitude_squared)

    # Next, we compute the polynomial modulation
    zeros = np.zeros_like(distances)
    modulation = np.where(distances < high_cutoff,
                          1 - 6.0 * np.power((distances/high_cutoff), 5)
                          + 15.0 * np.power((distances/high_cutoff), 4)
                          - 10.0 * np.power((distances/high_cutoff), 3),
                          zeros)
    modulation = np.expand_dims(modulation, axis=3)

    polynomial_cutoff_rbf_manual = modulation * gauss_manual

    # Map tiny values to zero
    polynomial_cutoff_rbf_manual = np.where(
        np.abs(polynomial_cutoff_rbf_manual) > polynomial_cutoff_rbf.tolerance,
        polynomial_cutoff_rbf_manual,
        np.zeros_like(polynomial_cutoff_rbf_manual)
    )

    # centers and output values need to be the same
    np.testing.assert_allclose(centers,
                               polynomial_cutoff_rbf.centers, rtol=1e-5)
    np.testing.assert_allclose(polynomial_cutoff_rbf_layer.numpy(),
                               polynomial_cutoff_rbf_manual, rtol=1e-5)


def test_polynomial_cutoff_rbf_distance_masking():
    # Makes sure that if a distance mask is used, the corresponding
    # expanded distances returned by PolynomialCutoffRBF are zero

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = torch.randn((frames, beads, beads - 1), dtype=torch.float64)
    # Define random parameters for the RBF
    n_gaussians = np.random.randint(5, 10)
    high_cutoff = np.random.uniform(5.0, 10.0)
    low_cutoff = np.random.uniform(0.0, 4.0)
    alpha = np.random.uniform(0.1, 1.0)

    neighbor_cutoff = np.abs(np.random.rand())
    neighbors, neighbor_mask = g.get_neighbors(distances,
                                               cutoff=neighbor_cutoff)

    # Calculate Gaussian expansion using the implemented layer
    polynomial_cutoff_rbf = PolynomialCutoffRBF(high_cutoff=high_cutoff,
                                                low_cutoff=low_cutoff,
                                                n_gaussians=n_gaussians,
                                                alpha=alpha,
                                                tolerance=1e-8)
    polynomial_cutoff_rbf_layer = polynomial_cutoff_rbf.forward(
        torch.tensor(distances),
        distance_mask=neighbor_mask)

    # Manually calculate expansion with numpy
    # First, we compute the centers and the scaling factors
    centers = np.linspace(np.exp(-high_cutoff), np.exp(-low_cutoff),
                          n_gaussians).astype(np.float64)
    beta = np.power(((2/n_gaussians) * (1-np.exp(-high_cutoff))), -2)

    # Next, we compute the gaussian portion
    exp_distances = np.exp(-alpha * np.expand_dims(distances, axis=3))
    magnitude_squared = np.power(exp_distances - centers, 2)
    gauss_manual = np.exp(-beta * magnitude_squared)

    # Next, we compute the polynomial modulation
    zeros = np.zeros_like(distances)
    modulation = np.where(distances < high_cutoff,
                          1 - 6.0 * np.power((distances/high_cutoff), 5)
                          + 15.0 * np.power((distances/high_cutoff), 4)
                          - 10.0 * np.power((distances/high_cutoff), 3),
                          zeros)
    modulation = np.expand_dims(modulation, axis=3)

    polynomial_cutoff_rbf_manual = modulation * gauss_manual

    # Map tiny values to zero
    polynomial_cutoff_rbf_manual = np.where(
        np.abs(polynomial_cutoff_rbf_manual) > polynomial_cutoff_rbf.tolerance,
        polynomial_cutoff_rbf_manual,
        np.zeros_like(polynomial_cutoff_rbf_manual)
    )
    polynomial_cutoff_rbf_manual = torch.tensor(
        polynomial_cutoff_rbf_manual) * neighbor_mask[:, :, :, None].double()

    np.testing.assert_array_almost_equal(polynomial_cutoff_rbf_layer.numpy(),
                                         polynomial_cutoff_rbf_manual.numpy())


def test_polynomial_cutoff_rbf_normalize():
    # Tests to make sure that the output of PolynomialCutoffRBF is properly
    # normalized if 'normalize_output' is specified as True

    # Distances need to have shape (n_batch, n_beads, n_neighbors)
    distances = np.random.randn(frames, beads, beads - 1).astype(np.float64)
    # Define random parameters for the polynomial_cutoff RBF
    n_gaussians = np.random.randint(5, 10)
    high_cutoff = np.random.uniform(5.0, 10.0)
    low_cutoff = np.random.uniform(0.0, 4.0)
    alpha = np.random.uniform(0.1, 1.0)

    # Calculate Gaussian expansion using the implemented layer
    polynomial_cutoff_rbf = PolynomialCutoffRBF(high_cutoff=high_cutoff,
                                                low_cutoff=low_cutoff,
                                                n_gaussians=n_gaussians,
                                                alpha=alpha,
                                                normalize_output=True,
                                                tolerance=1e-8)
    polynomial_cutoff_rbf_layer = polynomial_cutoff_rbf.forward(
        torch.tensor(distances))

    # Manually calculate expansion with numpy
    # First, we compute the centers and the scaling factors
    centers = np.linspace(np.exp(-high_cutoff), np.exp(-low_cutoff),
                          n_gaussians).astype(np.float64)
    beta = np.power(((2/n_gaussians) * (1-np.exp(-high_cutoff))), -2)

    # Next, we compute the gaussian portion
    exp_distances = np.exp(-alpha * np.expand_dims(distances, axis=3))
    magnitude_squared = np.power(exp_distances - centers, 2)
    gauss_manual = np.exp(-beta * magnitude_squared)

    # Next, we compute the polynomial modulation
    zeros = np.zeros_like(distances)
    modulation = np.where(distances < high_cutoff,
                          1 - 6.0 * np.power((distances/high_cutoff), 5)
                          + 15.0 * np.power((distances/high_cutoff), 4)
                          - 10.0 * np.power((distances/high_cutoff), 3),
                          zeros)
    modulation = np.expand_dims(modulation, axis=3)

    polynomial_cutoff_rbf_manual = modulation * gauss_manual

    # Map tiny values to zero
    polynomial_cutoff_rbf_manual = np.where(
        np.abs(polynomial_cutoff_rbf_manual) > polynomial_cutoff_rbf.tolerance,
        polynomial_cutoff_rbf_manual,
        np.zeros_like(polynomial_cutoff_rbf_manual)
    )

    # manually normalize the output
    polynomial_cutoff_rbf_manual /= np.sum(polynomial_cutoff_rbf_manual,
                                           axis=3)[:, :, :, None]

    # centers and output values need to be the same
    np.testing.assert_allclose(centers,
                               polynomial_cutoff_rbf.centers, rtol=1e-5)
    np.testing.assert_allclose(polynomial_cutoff_rbf_layer.numpy(),
                               polynomial_cutoff_rbf_manual, rtol=1e-5)


def test_polynomial_cutoff_rbf_zero_cutoff():
    # This test ensures that a choice of zero cutoff produces
    # a set of basis functions that all occupy the same center

    # First, we generate a polynomial_cutoff RBF layer with a random number
    # of gaussians and a cutoff of zero
    n_gaussians = np.random.randint(5, 10)
    cutoff = 0.0
    polynomial_cutoff_rbf = PolynomialCutoffRBF(n_gaussians=n_gaussians,
                                                high_cutoff=cutoff, low_cutoff=cutoff)
    # First we test to see that \beta is infinite
    np.testing.assert_equal(np.inf, polynomial_cutoff_rbf.beta)

    # Next we make a mock array of centers at 1.0
    centers = torch.linspace(
        np.exp(-cutoff), np.exp(-cutoff), n_gaussians, dtype=torch.float64)

    # Here, we test to see that centers are equal in this corner case
    np.testing.assert_equal(centers.numpy(),
                            polynomial_cutoff_rbf.centers.numpy())


def test_shifted_softplus():
    # Make sure shifted softplus activation is consistent with
    # manual calculation

    # Initialize random feature vector
    feature = torch.randn((frames, beads), dtype=torch.float64)

    ssplus = ShiftedSoftplus()
    # Shifted softplus has the following form:
    # y = \ln\left(1 + e^{-x}\right) - \ln(2)
    manual_output = np.log(1.0 + np.exp(feature.numpy())) - np.log(2.0)

    np.testing.assert_allclose(manual_output, ssplus(feature).numpy())
