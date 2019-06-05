# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network.nnet import CGnet, LinearLayer, ForceLoss, RepulsionLayer

from cgnet.feature import ProteinBackboneStatistics, ProteinBackboneFeature


# Random test data
x0 = torch.rand((50, 1), requires_grad=True)
slope = np.random.randn()
noise = 0.7*torch.rand((50, 1))
y0 = x0.detach()*slope + noise

# Random linear protein
num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)
coords = torch.randn((num_examples, num_beads, 3), requires_grad=True)
stats = ProteinBackboneStatistics(coords.detach().numpy())
bondsdict = stats.get_bond_constants()
#bonds = dict((k, bondsdict[k]) for k in [(i, i+1) for i in range(num_beads-1)])
repul_distances = [i for i in stats.distances if abs(i[0]-i[1]) > 2]


def test_linear_layer():
    """Tests LinearLayer function for bias logic and input/output size"""

    rand = np.random.randint(1, 11)
    layers = LinearLayer(1, rand, activation=None, bias=True)
    rands = 3 * [rand]
    biases = list(np.random.randint(2, size=3))
    for r, bias in zip(rands, biases):
        layers += LinearLayer(r, r, activation=nn.ReLU(),
                              bias=bool(bias), dropout=0.3)
    layers += LinearLayer(rand, 1, activation=None, bias=False)
    biases = [1] + biases + [0]
    linears = [l for l in layers if isinstance(l, nn.Linear)]
    for layer, bias in zip(linears, biases):
        if isinstance(layer, nn.Linear):
            if layer.bias is not None:
                np.testing.assert_equal(bool(layer.bias.data[0]), bool(bias))
            else:
                np.testing.assert_equal(bool(layer.bias), bool(bias))
    seq = nn.Sequential(*layers)
    y = seq(x0)

    np.testing.assert_equal(x0.size(), y.size())

def test_repulsion_layer():
    # Tests RepulsionLayer class for calculation and output size
    print(bondsdict)
    print(stats.distances)
    repulsion_potential = RepulsionLayer(repul_distances, excluded_volume=5.5,
                           exponent=6.0, descriptions=stats.descriptions,
                           feature_type='Distances')
    feat_layer = ProteinBackboneFeature()
    feat = feat_layer(coords)
    energy = repulsion_potential(feat[:, repulsion_potential.feat_idx])

    np.testing.assert_equal(energy.size(), (num_examples, 1))

def test_cgnet():
    """Tests CGnet class criterion attribute, architecture size, and network output size"""
    rand = np.random.randint(1, 10)
    arch = LinearLayer(1, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, 1, bias=True, activation=nn.Tanh())\

    model = CGnet(arch, ForceLoss())
    np.testing.assert_equal(len(arch), model.arch.__len__())
    np.testing.assert_equal(True, isinstance(model.criterion, ForceLoss))

    energy, force = model.forward(x0)
    np.testing.assert_equal(energy.size(), (x0.size()[0], 1))
    np.testing.assert_equal(force.size(), y0.size())


def test_linear_regression():
    """Comparison of CGnet with sklearn linear regression for linear force

    Notes
    -----
    This test is quite forgiving in comparing the sklearn/CGnet results
    for learning a linear force feild/quadratic potential because the decimal
    accuracy is set to one decimal point. It could be lower, but the test might
    then occassionaly fail due to stochastic reasons associated with the dataset
    and the limited training routine.

    """

    layers = LinearLayer(1, 15, activation=nn.Softplus(), bias=True)
    layers += LinearLayer(15, 15, activation=nn.Softplus(), bias=True)
    layers += LinearLayer(15, 1, activation=nn.Softplus(), bias=True)
    model = CGnet(layers, ForceLoss())
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)
    epochs = 35
    for i in range(epochs):
        optimizer.zero_grad()
        U, F = model.forward(x0)
        loss = model.criterion(F, y0)
        loss.backward()
        optimizer.step()
    loss = loss.data.numpy()

    # sklearn model for comparison
    x = x0.detach().numpy()
    y = y0.numpy()

    lrg = LinearRegression()
    reg = lrg.fit(x, y)
    y_pred = reg.predict(x)

    np.testing.assert_almost_equal(mse(y, y_pred), loss, decimal=1)
