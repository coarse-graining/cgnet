# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import CGnet, LinearLayer, ForceLoss
from cgnet.network import RepulsionLayer, HarmonicLayer, ZscoreLayer
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

# Prior variables
bondsdict = stats.get_bond_constants(flip_dict=True, zscores=True)
bonds = dict((k, bondsdict[k]) for k in [(i, i+1) for i in range(num_beads-1)])

repul_distances = [i for i in stats.descriptions['Distances']
                   if abs(i[0]-i[1]) > 2]
ex_vols = np.random.uniform(2, 8, len(repul_distances))
exps = np.random.randint(1, 6, len(repul_distances))
repul_dict = dict((index, {'ex_vol': ex_vol, 'exp': exp})
                  for index, ex_vol, exp
                  in zip(repul_distances, ex_vols, exps))

descriptions = stats.descriptions
nums = [len(descriptions['Distances']), len(descriptions['Angles']),
        len(descriptions['Dihedral_cosines']),
        len(descriptions['Dihedral_sines'])]
descs = [key for key in descriptions.keys()]
zscores = stats.get_zscores(tensor=True, as_dict=False).float()


def test_linear_layer():
    # Tests LinearLayer function for bias logic and input/output size

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


def test_zscore_layer():
    # Tests ZscoreLayer() for correct normalization

    # Notes
    # -----
    # rescaled_feat_truth is in principle equal to:
    # 
    # from sklearn.preprocessing import StandardScaler
    # scalar = StandardScaler()
    # rescaled_feat_truth = scalar.fit_transform(feat)
    # 
    # However, the equality is only preserved with precision >= 1e-4.

    feat_layer = ProteinBackboneFeature()
    feat = feat_layer(coords)
    rescaled_feat_truth = (feat - zscores[0, :])/zscores[1, :]

    zlayer = ZscoreLayer(zscores)
    rescaled_feat = zlayer(feat)

    np.testing.assert_array_equal(rescaled_feat.detach().numpy(),
                            rescaled_feat_truth.detach().numpy())


def test_repulsion_layer():
    # Tests RepulsionLayer class for calculation and output size

    repulsion_potential = RepulsionLayer(repul_dict,
                                         descriptions=descriptions,
                                         feature_type='Distances')
    feat_layer = ProteinBackboneFeature()
    feat = feat_layer(coords)
    energy = repulsion_potential(feat[:, repulsion_potential.feat_idx])

    np.testing.assert_equal(energy.size(), (num_examples, 1))
    start_idx = 0
    feat_idx = []
    for num, desc in zip(nums, descs):
        if 'Distances' == desc:
            break
        else:
            start_idx += num
    for pair in repul_distances:
        feat_idx.append(start_idx +
                        descriptions['Distances'].index(pair))
    p1 = torch.tensor(ex_vols).float()
    p2 = torch.tensor(exps).float()
    energy_check = torch.sum((p1/feat[:, feat_idx]) ** p2,
                             1).reshape(len(feat), 1) / 2
    np.testing.assert_equal(energy.detach().numpy(),
                            energy_check.detach().numpy())


def test_harmonic_layer():
    # Tests HarmonicLayer class for calculation and output size

    harmonic_potential = HarmonicLayer(bonds, descriptions=descriptions,
                                       feature_type='Distances')
    feat_layer = ProteinBackboneFeature()
    feat = feat_layer(coords)
    energy = harmonic_potential(feat[:, harmonic_potential.feat_idx])

    np.testing.assert_equal(energy.size(), (num_examples, 1))
    start_idx = 0
    feat_idx = []
    features = []
    harmonic_parameters = torch.tensor([])
    for num, desc in zip(nums, descs):
        if 'Distances' == desc:
            break
        else:
            start_idx += num
    for key, params in bonds.items():
        features.append(key)
        feat_idx.append(start_idx +
                        descriptions['Distances'].index(key))
        harmonic_parameters = torch.cat((harmonic_parameters,
                                         torch.tensor([[params['k']],
                                                       [params['mean']]])), dim=1)
    energy_check = torch.sum(harmonic_parameters[0, :] * (feat[:, feat_idx] -
                                                          harmonic_parameters[1, :]) ** 2,
                             1).reshape(len(feat), 1) / 2

    np.testing.assert_equal(energy.detach().numpy(),
                            energy_check.detach().numpy())


def test_cgnet():
    # Tests CGnet class criterion attribute, architecture size, and network
    # output size. Also tests prior embedding.

    harmonic_potential = HarmonicLayer(bonds, descriptions=stats.descriptions,
                                       feature_type='Distances')
    feature_layer = ProteinBackboneFeature()
    num_feats = feature_layer(coords).size()[1]

    rand = np.random.randint(1, 10)
    arch = LinearLayer(num_feats, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, 1, bias=True, activation=None)\

    model = CGnet(arch, ForceLoss(), feature=ProteinBackboneFeature(),
                  priors=[harmonic_potential])
    np.testing.assert_equal(True, model.priors is not None)
    np.testing.assert_equal(len(arch), model.arch.__len__())
    np.testing.assert_equal(True, isinstance(model.criterion, ForceLoss))
    np.testing.assert_equal(True, isinstance(model.arch, nn.Sequential))
    np.testing.assert_equal(True, isinstance(model.priors, nn.Sequential))

    energy, force = model.forward(coords)
    np.testing.assert_equal(energy.size(), (coords.size()[0], 1))
    np.testing.assert_equal(force.size(), coords.size())


def test_linear_regression():
    # Comparison of CGnet with sklearn linear regression for linear force

    # Notes
    # -----
    # This test is quite forgiving in comparing the sklearn/CGnet results
    # for learning a linear force feild/quadratic potential because the decimal
    # accuracy is set to one decimal point. It could be lower, but the test
    # might then occassionaly fail due to stochastic reasons associated with
    # the dataset and the limited training routine.

    layers = LinearLayer(1, 15, activation=nn.Softplus(), bias=True)
    layers += LinearLayer(15, 15, activation=nn.Softplus(), bias=True)
    layers += LinearLayer(15, 1, activation=nn.Softplus(), bias=True)
    model = CGnet(layers, ForceLoss())
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
