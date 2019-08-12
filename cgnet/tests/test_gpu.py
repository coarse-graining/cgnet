# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, ForceLoss,
                           RepulsionLayer, HarmonicLayer, ZscoreLayer)
from cgnet.feature import (GeometryStatistics, GeometryFeature,
                           MoleculeDataset, LinearLayer)
from torch.utils.data import DataLoader
from nose.exc import SkipTest


num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)
width = np.random.randint(2, high=10)


def test_model_gpu_mount():
    # Tests to see if forwarding through CGnet mounted on GPU runs properly
    if not torch.cuda.is_available():
        raise SkipTest("GPU not available for testing.")
    coords = np.random.randn(num_examples, num_beads, 3).astype('float32')
    forces = np.random.randn(num_examples, num_beads, 3).astype('float32')
    stats = GeometryStatistics(coords)
    feat_layer = GeometryFeature(feature_tuples=stats.feature_tuples)

    bonds_idx = stats.return_indices('Bonds')
    bonds_list = stats.get_prior_statistics(features='Bonds', as_list=True)

    repul_tuples = [i for i in stats.descriptions['Distances']
                       if abs(i[0]-i[1]) > 2]
    repul_idx = stats.return_indices(features=repul_tuples)
    ex_vols = np.random.uniform(2, 8, len(repul_idx))
    exps = np.random.randint(1, 6, len(repul_idx))
    repul_list = [{'ex_vol' : ex_vol, 'exp' : exp}
                  for ex_vol, exp in zip(ex_vols, exps)]

    zscores = stats.get_zscore_array()

    device = torch.device('cuda')
    dataset = MoleculeDataset(coords, forces, device=device)
    loader = DataLoader(dataset, batch_size=1)
    arch = [ZscoreLayer(zscores)]
    arch += LinearLayer(sum(nums), width)
    arch += LinearLayer(width, 1)

    priors = [HarmonicLayer(bonds_list, bonds_idx)]
    priors += [RepulsionLayer(repul_list, repul_idx)]

    model = CGnet(arch, ForceLoss(), feature=feat_layer,
                  priors=priors).float().to(device)
    for _, batch in enumerate(loader):
        coords, forces = batch
        pot, pred_force = model.forward(coords)

    np.testing.assert_equal(pot.device.type, 'cuda')
    np.testing.assert_equal(pred_force.device.type, 'cuda')
