# Authors: Nick Charron

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from cgnet.network import lipschitz_projection, dataset_loss
from cgnet.network import CGnet, ForceLoss, LinearLayer
from cgnet.feature import MoleculeDataset

coords = np.random.randn(10, 2).astype('float32')
forces = np.random.randn(10, 2).astype('float32')
dataset = MoleculeDataset(coords, forces)
sampler = SubsetRandomSampler(np.arange(0, 10, 1))
loader = DataLoader(dataset, sampler=sampler,
                    batch_size=np.random.randint(2, high=10))

width = np.random.randint(3, high=10)

arch = LinearLayer(2, 2, activation=nn.Tanh()) +\
    LinearLayer(2, 1, activation=None)

model = CGnet(arch, ForceLoss()).float()


def test_lipschitz():
    # Test hard lipschitz projection
    _lambda = float(1e-12)
    pre_projection_weights = []
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            pre_projection_weights.append(layer.weight.data)

    lipschitz_projection(model, _lambda)

    post_projection_weights = []
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            post_projection_weights.append(layer.weight.data)
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal, pre, post)

    # Test soft lipschitz projection
    _lambda = float(1e12)
    pre_projection_weights = []
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            pre_projection_weights.append(layer.weight.data)

    lipschitz_projection(model, _lambda)

    post_projection_weights = []
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            post_projection_weights.append(layer.weight.data)
    for pre, post in zip(pre_projection_weights, post_projection_weights):
        np.testing.assert_array_equal(pre, post)


def test_dataset_loss():
    # Test dataset loss by comparing results from different batch sizes
    # Batch size != 1
    loss = dataset_loss(model, loader)

    # Batch size = 1
    loader2 = DataLoader(dataset, sampler=sampler, batch_size=1)
    loss2 = dataset_loss(model, loader2)

    np.testing.assert_almost_equal(loss, loss2, decimal=5)
