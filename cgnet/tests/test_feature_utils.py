# Author: Dominik Lemm
# Contributors: Brooke Husic, Nick Charron

import numpy as np
import torch

from cgnet.feature.utils import RadialBasisFunction, ShiftedSoftplus
from cgnet.feature.statistics import GeometryStatistics
from cgnet.feature.feature import GeometryFeature


# Random protein
num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)
coords = np.random.randn(num_examples, num_beads, 3)
stats = GeometryStatistics(coords, get_redundant_distance_mapping=True)
feat = GeometryFeature(feature_tuples=stats.feature_tuples)

def test_radial_basis_function():
    # Make sure radial basis functions are consistent with manual calculation

    # Get distances from feature output and use grab redundant distance mapping
    output = feat(torch.tensor(coords, requires_grad=True).float())
    variance = np.random.random() + 1
    n_gaussians = np.random.randint(5, 10)
    cutoff = np.random.uniform(1.0, 5.0)
    # Calculate Gaussian expansion using the implemented layer
    rbf = RadialBasisFunction(stats.redundant_distance_mapping,
                              cutoff=cutoff, num_gaussians=n_gaussians,
                              variance=variance)
    gauss_layer = rbf.forward(output[:, rbf.redundant_callback_mapping])

    # Manually calculate expansion
    distances = output[:, stats.redundant_distance_mapping]
    centers = torch.linspace(0.0, cutoff, n_gaussians)
    coefficient = -0.5 / variance
    magnitude_squared = torch.pow(distances.unsqueeze(dim=3) - centers, 2)
    gauss_manual = torch.exp(coefficient * magnitude_squared)

    np.testing.assert_equal(centers.shape, rbf.centers.shape)
    np.testing.assert_allclose(gauss_layer.detach().numpy(), gauss_manual.detach().numpy())


def test_shifted_softplus():
    # Make sure shifted softplus activation is consistent with
    # manual calculation

    # Initialize random feature vector
    feature = torch.randn((num_examples, num_beads), dtype=torch.double)

    ssplus = ShiftedSoftplus()
    manual_output = torch.log(1.0 + torch.exp(feature)) - np.log(2.0)

    np.testing.assert_allclose(manual_output, ssplus(feature))
