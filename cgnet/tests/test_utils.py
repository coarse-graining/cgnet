# Authors: Nick Charron, Brooke Husic

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from cgnet.network import lipschitz_projection, dataset_loss, Simulation
from cgnet.network import CGnet, ForceLoss, LinearLayer
from cgnet.feature import MoleculeDataset

frames = np.random.randint(1, 3)
beads = np.random.randint(4, 10)
dims = 2

coords = np.random.randn(frames, beads, dims).astype('float32')
forces = np.random.randn(frames, beads, dims).astype('float32')
dataset = MoleculeDataset(coords, forces)
sampler = SubsetRandomSampler(np.arange(0, frames, 1))
loader = DataLoader(dataset, sampler=sampler,
                    batch_size=np.random.randint(2, high=10))

arch = (LinearLayer(dims, dims, activation=nn.Tanh()) +
        LinearLayer(dims, 1, activation=None))

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

    np.testing.assert_allclose(loss, loss2, rtol=1e-5)
