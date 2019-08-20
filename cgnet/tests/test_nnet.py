# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, ForceLoss, RepulsionLayer,
                           HarmonicLayer, ZscoreLayer, Simulation)
from cgnet.feature import (GeometryStatistics, GeometryFeature,
                           LinearLayer)

# The following sets up data for linear regession comparison test
x0 = torch.rand((50, 1), requires_grad=True)  # 50 1D input examples
slope = np.random.randn()  # Random slope for linear force
noise = 0.7*torch.rand((50, 1))  # Gaussian Noise for target
y0 = x0.detach()*slope + noise  # Noisy target

# The following sets up random linear protein data
frames = np.random.randint(10, 30)  # Number of frames
beads = np.random.randint(5, 10)  # Number of coarse-granined beads
dims = 3  # Number of dimensions

# Create mock linear protein simulation data and create statistics
coords = torch.randn((frames, beads, 3), requires_grad=True)
geom_stats = GeometryStatistics(coords.detach().numpy())


def test_linear_layer():
    # Tests LinearLayer function for bias logic and input/output size

    width = np.random.randint(1, 11)  # Random linear layer widths
    layers = LinearLayer(1, width, activation=None, bias=True)
    # Here, we create a simple 3 layer hidden architecture using the
    # random width defined above
    widths = 3 * [width]
    # Random bias binary mask for hidden layers
    biases = list(np.random.randint(2, size=3))
    # Assemble internal layers with random bias dropout
    for width, bias in zip(widths, biases):
        layers += LinearLayer(width, width, activation=nn.ReLU(),
                              bias=bool(bias), dropout=0.3)
    layers += LinearLayer(width, 1, activation=None, bias=True)
    # Here we extend the bias binary mask to the input and output layers
    # giving them default bias=True values
    biases = [1] + biases + [1]
    # Next, we isolate the nn.Linear modules and test if the
    # bias mask is respected
    linear_layer_list = [l for l in layers if isinstance(l, nn.Linear)]
    for layer, bias in zip(linear_layer_list, biases):
        if layer.bias is not None:
            np.testing.assert_equal(bool(layer.bias.data[0]), bool(bias))
        else:
            np.testing.assert_equal(bool(layer.bias), bool(bias))
    # Next, we test the equality of the output and input sizes
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

    # Complete prior dictionary
    full_prior_stats = geom_stats.get_prior_statistics()

    # First compute the reference zscore-rescaled features
    zscores = torch.zeros((2, len(full_prior_stats)))
    for i, key in enumerate(full_prior_stats.keys()):
        zscores[0, i] = full_prior_stats[key]['mean']
        zscores[1, i] = full_prior_stats[key]['std']

    # Then we create a feature layer and featurize our linear protein test data
    geom_feat = GeometryFeature(n_beads=beads)
    feat = geom_feat(coords)
    rescaled_feat_truth = (feat - zscores[0, :])/zscores[1, :]

    # Next, we instance a ZscoreLayer and test to see if its forward
    # method matches the reference calculation above
    zlayer = ZscoreLayer(zscores)
    rescaled_feat = zlayer(feat)

    np.testing.assert_array_equal(rescaled_feat.detach().numpy(),
                                  rescaled_feat_truth.detach().numpy())


def test_repulsion_layer():
    # Tests RepulsionLayer class for calculation and output size

    # First, we use the preamble repulsion variables to instance a
    # a RepulsionLayer. We pass the output of a feature layer to compare
    # RepulsionLayer forward method to manual energy calculation

    # The following sets up distance variables
    # List of distances at least 2 beads apart for RepulsionLayer tests
    repul_distances = [i for i in geom_stats.descriptions['Distances']
                       if abs(i[0]-i[1]) > 2]
    repul_idx = geom_stats.return_indices(repul_distances)  # Indices of beads
    # Random excluded volumes
    ex_vols = np.random.uniform(2, 8, len(repul_distances))
    # Random interaction exponentials
    exps = np.random.randint(1, 6, len(repul_distances))
    # List of interaction dictionaries for forming RepulsionLayers
    repul_list = [{'ex_vol': ex_vol, "exp": exp}
                  for ex_vol, exp in zip(ex_vols, exps)]

    repulsion_potential = RepulsionLayer(repul_idx, repul_list)
    geom_feat = GeometryFeature(n_beads=beads)
    output_features = geom_feat(coords)
    energy = repulsion_potential(output_features[:,
                                 repulsion_potential.callback_indices])

    # Test to see if RepulsionLayer ouput is scalar energy
    np.testing.assert_equal(energy.size(), (frames, 1))
    # Test to see if RepulsionLayer callback_indices are correct
    assert repul_idx == repulsion_potential.callback_indices
    # Next, we test to see if the manually calculated energy
    # matches the output of the RepulsionLayer
    p1 = torch.tensor(ex_vols).float()
    p2 = torch.tensor(exps).float()
    energy_check = torch.sum((p1/output_features[:, repul_idx]) ** p2,
                             1).reshape(len(output_features), 1) / 2
    np.testing.assert_array_equal(energy.detach().numpy(),
                                  energy_check.detach().numpy())


def test_harmonic_layer():
    # Tests HarmonicLayer class for calculation and output size

    # Set up bond indices (integers) and interactiosn
    bonds_idx = geom_stats.return_indices('Bonds')  # Bond indices
    # List of bond interaction dictionaries for assembling priors
    bonds_interactions, _ = geom_stats.get_prior_statistics(
        features='Bonds', as_list=True)

    # First, we use the preamble bond variable to instance a
    # HarmonicLayer. We pass the output of a feature layer to compare
    # HarmonicLayer forward method to manual energy calculation
    harmonic_potential = HarmonicLayer(bonds_idx, bonds_interactions)
    geom_feat = GeometryFeature(n_beads=beads)
    output_features = geom_feat(coords)
    energy = harmonic_potential(output_features[:,
                                 harmonic_potential.callback_indices])

    # Test to see if HarmonicLayer output is scalar energy
    np.testing.assert_equal(energy.size(), (frames, 1))
    # Test to see if HarmonicLayer callback_indices are correct
    assert bonds_idx == harmonic_potential.callback_indices

    # Next, we test to see if the manually calculated energy
    # matches the output of the HarmonicLayer
    feature_stats = geom_stats.get_prior_statistics('Bonds')
    harmonic_parameters = torch.tensor([])
    for bead_tuple, stat in feature_stats.items():
        harmonic_parameters = torch.cat((
            harmonic_parameters,
            torch.tensor([[stat['k']],
                          [stat['mean']]])), dim=1)
    energy_check = torch.sum(harmonic_parameters[0, :] *
                             (output_features[:, bonds_idx] -
                             harmonic_parameters[1, :]) ** 2,
                             1).reshape(len(output_features), 1) / 2

    np.testing.assert_array_equal(energy.detach().numpy(),
                                  energy_check.detach().numpy())


def test_prior_callback_order_1():
    # Tests the order of prior callbacks with respect to feature layer output
    # First, we instance a statistics object
    stats = GeometryStatistics(coords)

    # Next, we isolate the bonds from the distance feature tuples
    bonds_tuples = [beads for beads in stats.master_description_tuples
                    if len(beads) == 2 and abs(beads[0] - beads[1]) == 1]

    # Shuffle bonds and get shuffled indices
    np.random.shuffle(bonds_tuples)
    bonds_idx = stats.return_indices(bonds_tuples)

    # Next, we create a shuffled harmonic potential using shuffled statistics
    bonds_interactions, _ = stats.get_prior_statistics(features=list(bonds_tuples),
                                                       as_list=True)
    harmonic_potential = HarmonicLayer(bonds_idx, bonds_interactions)

    # Test to see if callback indices are correctly embedded
    np.testing.assert_array_equal(
        bonds_idx, harmonic_potential.callback_indices)


def test_prior_callback_order_2():
    # The order of callback indices should not change the final HarmonicLayer
    # energy output - so here we test to see if the shuffled HarmonicLayer output
    # matches a manual calculation using the default order
    # We use the same setup as in test_prior_callback_order_1, but add further
    # add a GeometryFeature layer for comparing energy outputs
    stats = GeometryStatistics(coords)
    geom_feat = GeometryFeature(n_beads=beads)
    output_features = geom_feat(coords)
    bonds_tuples = [beads for beads in stats.master_description_tuples
                    if len(beads) == 2 and abs(beads[0] - beads[1]) == 1]
    np.random.shuffle(bonds_tuples)
    bonds_idx = stats.return_indices(bonds_tuples)
    bonds_interactions, _ = stats.get_prior_statistics(features=list(bonds_tuples),
                                                       as_list=True)
    harmonic_potential = HarmonicLayer(bonds_idx, bonds_interactions)

    # Here, we test the energy of the shuffled HarmonicLayer with a manual
    # calculation according to the default GeometryStatistics bond order
    energy = harmonic_potential(output_features[:,
                                harmonic_potential.callback_indices])
    feature_stats = stats.get_prior_statistics('Bonds')
    feature_idx = stats.return_indices('Bonds')
    harmonic_parameters = torch.tensor([])
    for bead_tuple, stat in feature_stats.items():
        harmonic_parameters = torch.cat((
            harmonic_parameters,
            torch.tensor([[stat['k']],
                          [stat['mean']]])), dim=1)
    energy_check = torch.sum(harmonic_parameters[0, :] *
                             (output_features[:, feature_idx] -
                             harmonic_parameters[1, :]) ** 2,
                             1).reshape(len(output_features), 1) / 2
    np.testing.assert_allclose(np.sum(energy.detach().numpy()),
                               np.sum(energy_check.detach().numpy()), rtol=1e-4)


def test_prior_with_stats_dropout():
    # Test the order of prior callbacks when the statistices are missing one
    # of the four default backbone features: 'Distances', 'Angles',
    # 'Dihedral_cosines', and 'Dihedral_sines,'

    # First, we determine a shuffled, random feature mask
    # and create an instance of GeometryStatistics
    feature_bools = [1] + [np.random.randint(0, high=1) for _ in range(2)]
    np.random.shuffle(feature_bools)
    stats = GeometryStatistics(coords,
                               get_all_distances=feature_bools[0],
                               get_backbone_angles=feature_bools[1],
                               get_backbone_dihedrals=feature_bools[2])

    # Here we construct priors on available features and test the callback order
    if 'Distances' in stats.descriptions:
        # HarmonicLayer bonds test with random constants & means
        bonds_interactions, _ = stats.get_prior_statistics(features='Bonds',
                                                           as_list=True)
        bonds_idx = stats.return_indices('Bonds')
        harmonic_potential = HarmonicLayer(bonds_idx, bonds_interactions)
        np.testing.assert_array_equal(bonds_idx,
                                      harmonic_potential.callback_indices)

        # RepulsionLayer test with random exclusion volumess & exponents
        repul_distances = stats.descriptions['Distances']
        dist_idx = stats.return_indices('Distances')
        ex_vols = np.random.uniform(2, 8, len(repul_distances))
        exps = np.random.randint(1, 6, len(repul_distances))
        repul_interactions = [{'ex_vol': ex_vol, "exp": exp} for ex_vol, exp
                              in zip(ex_vols, exps)]
        repulsion_potential = RepulsionLayer(dist_idx, repul_interactions)
        np.testing.assert_array_equal(dist_idx,
                                      repulsion_potential.callback_indices)
    # Next, we run the same tests as above but for non-distance features
    # using harmonic priors
    for name in ['Angles', 'Dihedral_cosines', 'Dihedral_sines']:
        if name in stats.descriptions:
            feat_interactions, _ = stats.get_prior_statistics(features=name,
                                                              as_list=True)
            feat_idx = stats.return_indices(name)
            harmonic_potential = HarmonicLayer(feat_idx, feat_interactions)
            np.testing.assert_array_equal(feat_idx,
                                          harmonic_potential.callback_indices)


def test_cgnet():
    # Tests CGnet class criterion attribute, architecture size, and network
    # output size. Also tests priors for proper residual connection to
    # feature layer.

    # First, we set up a bond harmonic prior and a GeometryFeature layer
    bonds_idx = geom_stats.return_indices('Bonds')
    bonds_interactions, _ = geom_stats.get_prior_statistics(
        features='Bonds', as_list=True)
    harmonic_potential = HarmonicLayer(bonds_idx, bonds_interactions)
    feature_layer = GeometryFeature(n_beads=beads)
    num_feats = feature_layer(coords).size()[1]

    # Next, we create a 4 layer hidden architecture with a random width
    # and with a scalar output
    rand = np.random.randint(1, 10)
    arch = (LinearLayer(num_feats, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, 1, bias=True, activation=None))

    # Next, we instance a CGnet model using the above objects
    # with force matching as a loss criterion
    model = CGnet(arch, ForceLoss(), feature=feature_layer,
                  priors=[harmonic_potential])

    # Test to see if the prior is embedded
    np.testing.assert_equal(True, model.priors is not None)

    # Test to see if the hidden architexture has the correct length
    np.testing.assert_equal(len(arch), model.arch.__len__())

    # Test to see if criterion is embedded correctly
    np.testing.assert_equal(True, isinstance(model.criterion, ForceLoss))

    # Next, we forward the test protein data from the preamble through
    # the model
    energy, force = model.forward(coords)
    # Here, we test to see if the predicted energy is scalar
    # and the predicted forces are the same dimension as the input coordinates
    np.testing.assert_equal(energy.size(), (coords.size()[0], 1))
    np.testing.assert_equal(force.size(), coords.size())


def test_cgnet_simulation():
    # Tests a simulation from a CGnet built with the GeometryFeature
    # for the shapes of its coordinate, force, and potential outputs

    # First, we set up a bond harmonic prior and a GeometryFeature layer
    bonds_idx = geom_stats.return_indices('Bonds')
    bonds_interactions, _ = geom_stats.get_prior_statistics(
        features='Bonds', as_list=True)
    harmonic_potential = HarmonicLayer(bonds_idx, bonds_interactions)
    feature_layer = GeometryFeature(n_beads=beads)
    num_feats = feature_layer(coords).size()[1]

    # Next, we create a 4 layer hidden architecture with a random width
    # and with a scalar output
    rand = np.random.randint(1, 10)
    arch = (LinearLayer(num_feats, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, rand, bias=True, activation=nn.Tanh())
            + LinearLayer(rand, 1, bias=True, activation=None))

    # Next, we instance a CGnet model using the above objects
    # with force matching as a loss criterion
    model = CGnet(arch, ForceLoss(), feature=feature_layer,
                  priors=[harmonic_potential])

    # Here, we produce mock target protein force data
    forces = torch.randn((frames, beads, 3), requires_grad=False)

    # Here, we create an optimizer for traning the model,
    # and we train it for one epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)
    optimizer.zero_grad()
    energy, pred_forces = model.forward(coords)
    loss = model.criterion(pred_forces, forces)
    loss.backward()
    optimizer.step()

    # Here, we define random simulation frame lengths
    # as well as randomly choosing to save every 2 or 4 frames
    length = np.random.choice([2, 4])*2
    save = np.random.choice([2, 4])

    # Here we instance a simulation class and produce a CG trajectory
    my_sim = Simulation(model, coords, beta=geom_stats.beta, length=length,
                        save_interval=save, save_forces=True,
                        save_potential=True)

    traj = my_sim.simulate()

    # We test to see if the trajectory is the proper shape based on the above
    # choices for simulation length and frame saving
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

    # First, we instance a CGnet model 2 layers deep and 15 nodes wide
    layers = LinearLayer(1, 15, activation=nn.Softplus(), bias=True)
    layers += LinearLayer(15, 15, activation=nn.Softplus(), bias=True)
    layers += LinearLayer(15, 1, activation=nn.Softplus(), bias=True)
    model = CGnet(layers, ForceLoss())

    # Next, we define the optimizer and train for 35 epochs on the test linear
    # regression data defined in the preamble
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)
    epochs = 35
    for i in range(epochs):
        optimizer.zero_grad()
        energy, force = model.forward(x0)
        loss = model.criterion(force, y0)
        loss.backward()
        optimizer.step()
    loss = loss.data.numpy()

    # We produce numpy verions of the training data
    x = x0.detach().numpy()
    y = y0.numpy()

    # Here, we instance an sklearn linear regression model for comparison to
    # CGnet
    lrg = LinearRegression()
    reg = lrg.fit(x, y)
    y_pred = reg.predict(x)

    # Here, we test to to see if MSE losses are close up to a tolerance.
    np.testing.assert_almost_equal(mse(y, y_pred), loss, decimal=1)
