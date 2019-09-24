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
                           MoleculeDataset, LinearLayer, SchnetFeature,
                           FeatureCombiner)
from torch.utils.data import DataLoader
from nose.exc import SkipTest

num_examples = np.random.randint(10, 30)
num_beads = np.random.randint(5, 10)
width = np.random.randint(2, high=10)

def test_cgnet_mount_method():
    # This test asseses the device mounting method for CGnet
    # First we create a random data set of a mock linear protein
    coords = np.random.randn(num_examples, num_beads, 3).astype('float32')
    forces = np.random.randn(num_examples, num_beads, 3).astype('float32')
    # Next, we gather the statistics for Bond/Repulsion priors
    stats = GeometryStatistics(coords)

    bonds_list, _ = stats.get_prior_statistsics('Bonds', as_list=True)
    bonds_idx = stats.return_indices('Bonds')

    repul_tuples = [i for i in stats.descriptions['Distances']
                       if abs(i[0]-i[1]) > 2]
    repul_idx = stats.return_indices(features=repul_distances)
    ex_vols = np.random.uniform(2, 8, len(repul_distances))
    exps = np.random.randint(1, 6, len(repul_distances))
    repul_list = [{'ex_vol' : ex_vol, 'exp' : exp}
                  for ex_vol, exp in zip(ex_vols, exps)]
    # Next, we also grab the Zscores
    zscores = stats.get_zscore_array()

    # Next, we define our local CUDA device
    device = torch.device('cuda')

    # Next, we create the hidden architecture of CGnet
    arch = [ZscoreLayer(zscores)]
    arch += LinearLayer(len(stats.master_description_tuples), width)
    arch += LinearLayer(width, 1)

    # Here, we assemble the priors list
    priors = [HarmonicLayer(bonds_list, bonds_idx)]
    priors += [RepulsionLayer(repul_list, repul_idx)]

    # Next, we assemble a SchnetFeature with random initialization arguments
    feature_size = np.random.randint(5, high=10)  # random feature size
    n_embeddings = np.random.randint(3, high=5)  # random embedding number
    embedding_dim = feature_size  # embedding property size
    n_interaction_blocks = np.random.randint(
        1, 3)  # random number of interactions
    neighbor_cutoff = np.random.uniform(0, 1)  # random neighbor cutoff
    # random embedding property
    embedding_property = torch.randint(low=0, high=n_embeddings,
                                       size=(n_frames, n_beads))
    embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings,
                                      embedding_dim=embedding_dim)
    schnet_feature = SchnetFeature(feature_size=feature_size,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=n_interaction_blocks,
                                   calculate_geometry=False,
                                   n_beads=n_beads,
                                   neighbor_cutoff=neighbor_cutoff)
    # Here we create a GeometryFeature, and we assemble our feates and
    # ZscoreLayer into a FeatureCombiner
    geometry_layer = GeometryFeature(feature_tuples=stats.feature_tuples)
    combined_features = FeatureCombiner([geometry_feature, schnet_feature])

    # Finally, we instance a CGnet model and mount it to the device
    model = CGnet(arch, ForceLoss(), feature=combined_features,
                  priors=priors).float().to(device)
    model.mount(device)

    # Next, we check to see if each layer is mounted correctly
    # This is done by checking if parameters/buffers are mapped to the correct
    # device, or that feature classes are imbued with the appropriate device 
    # First, we check features
    for layer in model.feature.layer_list:
       if isinstance(layer, (GeometryFeature, SchnetFeature):
           assert layer.device == device
       if isinstance(layer, ZscoreLayer):
           assert layer.zscores.device == device
    # Next, we check priors
    for prior in model.priors:
        if isinstance(prior, HarmonicLayer):
            assert prior.harmonic_parameters.device == device
        if isinstance(prior, RepulsionLayer):
            assert prior.repulsion_parameters.device == device
    # Finally, we check the arch layers
    for param in model.paramters():
        assert param.device == device

    # Lastly, we perform a forward pass over the data and
    coords.to(device)
    pot, pred_force = model.forward(coords)
    assert pot.device == device
    assert pred_force.device == device
