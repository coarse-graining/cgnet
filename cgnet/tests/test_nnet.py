import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from network.nnet import *

# Random test data
x0 = torch.rand((3000, 2), requires_grad=True)
y0 = torch.rand((3000, 2))
batch = {'traj': x0, 'force': y0}


def test_linear_layer():
    """Tests LinearLayer function"""

    rand = np.random.randint(1, 11)
    layers = LinearLayer(2, rand, activation=None, bias=True)
    rands = 3 * [rand]
    biases = np.random.randint(2, size=3)
    for r, bias in zip(rands, baises):
        layers += LinearLayer(r, r, activation=nn.ReLU(),
                              bias=bias, dropout=0.3)
    layers += LinearLayer(rand, 2, activation=None)
    for layer in layers:
        if isinstance(layer, type(nn.Linear())):
            np.testing.assert_equal(bool(layer.bias), bool(bias))

    seq = nn.Sequential(*layers)
    y = seq(x0)

    np.testing.assert_equal(x0.size(), y.size())


def test_net():
    """Tests Net class"""
    rand = np.random.randint(10)
    arch = LinearLayer(2, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, 1, bias=True, activation=nn.Tanh())\

    model = Net(arch, ForceLoss())
    np.testing.assert_equal(len(arch), model.arch.__len__())
    np.testing.assert_equal(isinstance(ForceLoss(), type(nn.Module())),
                            isinstance(model.criterion, type(nn.Module())))

    energy, force = model.forward(x0)
    np.testing.assert_equal(energy.size(), (x0.size()[0], 1))
    np.testing.assert_equal(force.size(), y0.size())


def test_linear_regression():
    """Comparison of single layer network for linear regression with sklearn"""

    layers = LinearLayer(2, 1, activation=None, bias=True)
    model = Net(layers, ForceLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = 500
    for i in range(epochs):
        optimizer.zero_grad()
        U, F = model.forward(x0)
        loss = model.criterion(F, y0)
        loss.backward()
        optimizer.step()
    loss = loss.data.numpy()

    # sklearn model for comparison

    lrg = LinearRegression()
    x = x0.detach().numpy()
    y = y0.numpy()
    reg = lrg.fit(x, y)
    y_pred = reg.predict(x)

    np.testing.assert_almost_equal(mse(y, y_pred), loss, decimal=3)
