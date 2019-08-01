# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, ForceLoss, RepulsionLayer,
                           HarmonicLayer, ZscoreLayer, Simulation,
                           assemble_harmonic_inputs)
from cgnet.feature import (GeometryStatistics, GeometryFeature,
                           LinearLayer)

# Random test data
x0 = torch.rand((50, 1), requires_grad=True)
slope = np.random.randn()
noise = 0.7*torch.rand((50, 1))
y0 = x0.detach()*slope + noise

# Random linear protein
frames = np.random.randint(10, 30)
beads = np.random.randint(5, 10)
dims = 3

coords = torch.randn((frames, beads, 3), requires_grad=True)
stats = GeometryStatistics(coords.detach().numpy())

# Prior variables
full_prior_stats = stats.get_prior_statistics()
bonds_stats = stats.get_prior_statistics(features='Bonds')
bonds_idx = stats.return_indices('Bonds')
bonds_dict = assemble_harmonic_inputs(bonds_stats, bonds_idx)

repul_distances = [i for i in stats.descriptions['Distances']
                   if abs(i[0]-i[1]) > 2]
repul_idx = stats.return_indices(repul_distances)

ex_vols = np.random.uniform(2, 8, len(repul_distances))
exps = np.random.randint(1, 6, len(repul_distances))
repul_dict = dict((idx, {'beads': beads,
                         'params': {'ex_vol': ex_vol, 'exp': exp}})
                  for idx, beads, ex_vol, exp
                  in zip(repul_idx, repul_distances, ex_vols, exps))

descriptions = stats.descriptions
order = stats.order
nums = [len(descriptions[desc]) for desc in order]
zscores = torch.zeros((2, len(full_prior_stats)))
for i, key in enumerate(full_prior_stats.keys()):
    zscores[0, i] = full_prior_stats[key]['mean']
    zscores[1, i] = full_prior_stats[key]['std']


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

    feat_layer = GeometryFeature(n_beads=beads)
    feat = feat_layer(coords)
    rescaled_feat_truth = (feat - zscores[0, :])/zscores[1, :]

    zlayer = ZscoreLayer(zscores)
    rescaled_feat = zlayer(feat)

    np.testing.assert_array_equal(rescaled_feat.detach().numpy(),
                                  rescaled_feat_truth.detach().numpy())


def test_repulsion_layer():
    # Tests RepulsionLayer class for calculation and output size

    repulsion_potential = RepulsionLayer(repul_dict)
    feat_layer = GeometryFeature(n_beads=beads)
    feat = feat_layer(coords)
    energy = repulsion_potential(feat[:, repulsion_potential.feat_idx])

    np.testing.assert_equal(energy.size(), (frames, 1))
    start_idx = 0
    feat_idx = []
    for num, desc in zip(nums, order):
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
    np.testing.assert_array_equal(energy.detach().numpy(),
                                  energy_check.detach().numpy())


def test_harmonic_layer():
    # Tests HarmonicLayer class for calculation and output size

    harmonic_potential = HarmonicLayer(bonds_dict)
    feat_layer = GeometryFeature(n_beads=beads)
    feat = feat_layer(coords)
    energy = harmonic_potential(feat[:, harmonic_potential.feat_idx])

    np.testing.assert_equal(energy.size(), (frames, 1))
    start_idx = 0
    feat_idx = stats.return_indices('Bonds')
    features = [stats.master_description_tuples[i] for i in feat_idx]
    assert features == harmonic_potential.features
    assert feat_idx == harmonic_potential.feat_idx

    feature_stats = stats.get_prior_statistics('Bonds')
    harmonic_parameters = torch.tensor([])
    for bead_tuple, stat in feature_stats.items():
        harmonic_parameters = torch.cat((
            harmonic_parameters,
            torch.tensor([[stat['k']],
                          [stat['mean']]])), dim=1)
    energy_check = torch.sum(harmonic_parameters[0, :] * (feat[:, feat_idx] -
                             harmonic_parameters[1, :]) ** 2,
                             1).reshape(len(feat), 1) / 2

    np.testing.assert_array_equal(energy.detach().numpy(),
                                  energy_check.detach().numpy())


def test_prior_callback_order():
    # Tests the order of prior callbacks with respect to feature layer output
    stats = GeometryStatistics(coords)
    feat_layer = GeometryFeature(n_beads=beads)
    feat = feat_layer(coords)
    bonds_tuples = [beads for beads in stats.master_description_tuples
                    if len(beads) == 2 and abs(beads[0] - beads[1]) == 1]
    np.random.shuffle(bonds_tuples)
    bonds_idx = stats.return_indices(bonds_tuples)
    bonds_stats = stats.get_prior_statistics(features=list(bonds_tuples))
    bonds_dict = assemble_harmonic_inputs(bonds_stats, bonds_idx)

    harmonic_potential = HarmonicLayer(bonds_dict)
    np.testing.assert_array_equal(bonds_idx, harmonic_potential.feat_idx)

    energy = harmonic_potential(feat[:, harmonic_potential.feat_idx])
    feature_stats = stats.get_prior_statistics('Bonds')
    feat_idx = stats.return_indices('Bonds')
    harmonic_parameters = torch.tensor([])
    for bead_tuple, stat in feature_stats.items():
        harmonic_parameters = torch.cat((
            harmonic_parameters,
            torch.tensor([[stat['k']],
                          [stat['mean']]])), dim=1)
    energy_check = torch.sum(harmonic_parameters[0, :] * (feat[:, feat_idx] -
                             harmonic_parameters[1, :]) ** 2,
                             1).reshape(len(feat), 1) / 2
    np.testing.assert_allclose(energy.detach().numpy(),
                               energy_check.detach().numpy(), rtol=1e-4)
    np.testing.assert_allclose(np.sum(energy.detach().numpy()),
                               np.sum(energy_check.detach().numpy()), rtol=1e-4)


def test_prior_with_stats_dropout():
    # Test the order of prior callbacks when the statistices are missing one
    # of the four default backbone features: 'Distances', 'Angles',
    # 'Dihedral_cosines', and 'Dihedral_sines,'

    # determined by random.
    feature_bools = [1] + [np.random.randint(0, high=1) for _ in range(2)]
    np.random.shuffle(feature_bools)
    stats = GeometryStatistics(coords,
                               get_all_distances=feature_bools[0],
                               get_backbone_angles=feature_bools[1],
                               get_backbone_dihedrals=feature_bools[2])

    feat_layer = GeometryFeature(feature_tuples=stats.feature_tuples)
    if 'Distances' in stats.descriptions:
        # HarmonicLayer bonds test with random constants & means
        bonds_stats = stats.get_prior_statistics(features='Bonds')
        bonds_idx = stats.return_indices('Bonds')
        bonds_dict = assemble_harmonic_inputs(bonds_stats, bonds_idx)
        harmonic_potential = HarmonicLayer(bonds_dict)
        np.testing.assert_array_equal(bonds_idx, harmonic_potential.feat_idx)

        # RepulsionLayer test with random exculsion vols & exps
        repul_distances = stats.descriptions['Distances']
        dist_idx = stats.return_indices('Distances')
        ex_vols = np.random.uniform(2, 8, len(repul_distances))
        exps = np.random.randint(1, 6, len(repul_distances))
        repul_dict = dict((index, {'beads': beads,
                                   'params': {'ex_vol': ex_vol, 'exp': exp}})
                          for index, beads, ex_vol, exp
                          in zip(dist_idx, repul_distances, ex_vols, exps))
        repulsion_potential = RepulsionLayer(repul_dict)
        np.testing.assert_array_equal(dist_idx, repulsion_potential.feat_idx)
        for name in ['Angles', 'Dihedral_cosines', 'Dihedral_sines']:
            if name in stats.descriptions:
                feat_stats = stats.get_prior_statistics(features=name)
                feat_idx = stats.return_indices(name)
                feat_dict = assemble_harmonic_inputs(feat_stats, feat_idx)
                harmonic_potential = HarmonicLayer(feat_dict)
                np.testing.assert_array_equal(feat_idx, harmonic_potential.feat_idx)

def test_cgnet():
    # Tests CGnet class criterion attribute, architecture size, and network
    # output size. Also tests prior embedding.
    harmonic_potential = HarmonicLayer(bonds_dict)
    feature_layer = GeometryFeature(n_beads=beads)
    num_feats = feature_layer(coords).size()[1]

    rand = np.random.randint(1, 10)
    arch = LinearLayer(num_feats, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, 1, bias=True, activation=None)\

    model = CGnet(arch, ForceLoss(), feature=feature_layer,
                  priors=[harmonic_potential])
    np.testing.assert_equal(True, model.priors is not None)
    np.testing.assert_equal(len(arch), model.arch.__len__())
    np.testing.assert_equal(True, isinstance(model.criterion, ForceLoss))
    np.testing.assert_equal(True, isinstance(model.arch, nn.Sequential))
    np.testing.assert_equal(True, isinstance(model.priors, nn.Sequential))

    energy, force = model.forward(coords)
    np.testing.assert_equal(energy.size(), (coords.size()[0], 1))
    np.testing.assert_equal(force.size(), coords.size())


def test_cgnet_simulation():
    # Tests a simulation from a CGnet built with the GeometryFeature
    # for the shapes of its coordinate, force, and potential outputs

    harmonic_potential = HarmonicLayer(bonds_dict)
    feature_layer = GeometryFeature(n_beads=beads)
    num_feats = feature_layer(coords).size()[1]

    rand = np.random.randint(1, 10)
    arch = LinearLayer(num_feats, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())\
        + LinearLayer(rand, 1, bias=True, activation=None)\

    model = CGnet(arch, ForceLoss(), feature=feature_layer,
                  priors=[harmonic_potential])

    forces = torch.randn((frames, beads, 3), requires_grad=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)
    optimizer.zero_grad()
    U, F = model.forward(coords)
    loss = model.criterion(F, forces)
    loss.backward()
    optimizer.step()

    length = np.random.choice([2, 4])*2
    save = np.random.choice([2, 4])
    my_sim = Simulation(model, coords, beta=stats.beta, length=length,
                        save_interval=save, save_forces=True,
                        save_potential=True)

    traj = my_sim.simulate()
    assert traj.shape == (frames, length // save, beads, dims)
    assert my_sim.simulated_forces.shape == (
        frames, length // save, beads, dims)
    assert my_sim.simulated_potential.shape == (frames, length // save, 1)


def test_linear_regression():
    # Comparison of CGnet with sklearn linear regression for linear force

    # Notes
    # -----
    # This test is quite forgiving in comparing the sklearn/CGnet results
    # for learning a linear force feild/quadratic potential because the decimal
    # accuracy is set to one decimal point. It could be lower, but the test
    # might then occassionaly fail due to stochastic reasons associated with
    # the dataset and the limited training routine.
    #
    # For this reason, we use np.testing.assert_almost_equal instead of
    # np.testing.assert_allclose

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
