# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, LinearLayer, ForceLoss,
                           RepulsionLayer, HarmonicLayer, ZscoreLayer)
from cgnet.feature import (ProteinBackboneStatistics, ProteinBackboneFeature,
                           MoleculeDataset)
from torch.utils.data import DataLoader
from nose.exc import SkipTest


num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)
width = np.random.randint(2, high=10)


def test_model_gpu_mount():
    if not torch.cuda.is_available():
        raise SkipTest("GPU not available for testing.")
    coords = np.random.randn(num_examples, num_beads, 3).astype('float32')
    forces = np.random.randn(num_examples, num_beads, 3).astype('float32')
    stats = ProteinBackboneStatistics(coords)

    bondsdict = stats.get_bond_constants(flip_dict=True, zscores=True)
    bonds = dict((k, bondsdict[k])
                 for k in [(i, i+1) for i in range(num_beads-1)])

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

    device = torch.device('cuda')
    dataset = MoleculeDataset(coords, forces, device=device)
    loader = DataLoader(dataset, batch_size=1)
    arch = [ZscoreLayer(zscores)]
    arch += LinearLayer(sum(nums), width)
    arch += LinearLayer(width, 1)

    priors = [HarmonicLayer(bonds, descriptions, "Distances")]
    priors += [RepulsionLayer(repul_dict, descriptions,
                              "Distances")]

    model = CGnet(arch, ForceLoss(), feature=ProteinBackboneFeature(),
                  priors=priors).float().to(device)
    for _, batch in enumerate(loader):
        coords, forces = batch
        pot, pred_force = model.forward(coords)

    np.testing.assert_equal(pot.device.type, 'cuda')
    np.testing.assert_equal(pred_force.device.type, 'cuda')
