# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import tempfile
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, ForceLoss,
                           RepulsionLayer, HarmonicLayer, ZscoreLayer)
from cgnet.feature import (GeometryStatistics, GeometryFeature,
                           MoleculeDataset, LinearLayer, SchnetFeature,
                           FeatureCombiner, CGBeadEmbedding)
from torch.utils.data import DataLoader
from nose.exc import SkipTest


def generate_model():
    # Generate random CGnet model and coordinates
    n_frames = np.random.randint(10, 30)
    n_beads = np.random.randint(5, 10)
    width = np.random.randint(2, high=10)

    # First we create a random data set of a mock linear protein
    coords = np.random.randn(n_frames, n_beads, 3).astype('float32')

    # Next, we gather the statistics for Bond/Repulsion priors
    stats = GeometryStatistics(coords, backbone_inds='all',
                               get_all_distances=True, get_backbone_angles=True,
                               get_backbone_dihedrals=True)

    bonds_list, _ = stats.get_prior_statistics('Bonds', as_list=True)
    bonds_idx = stats.return_indices('Bonds')

    repul_distances = [i for i in stats.descriptions['Distances']
                       if abs(i[0]-i[1]) > 2]
    repul_idx = stats.return_indices(features=repul_distances)
    ex_vols = np.random.uniform(2, 8, len(repul_distances))
    exps = np.random.randint(1, 6, len(repul_distances))
    repul_list = [{'ex_vol': ex_vol, 'exp': exp}
                  for ex_vol, exp in zip(ex_vols, exps)]
    # Next, we also grab the Zscores
    zscores, _ = stats.get_zscore_array()

    # Here, we assemble the priors list
    priors = [HarmonicLayer(bonds_idx, bonds_list)]
    priors += [RepulsionLayer(repul_idx, repul_list)]

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
    distance_idx = stats.return_indices("Distances")
    geometry_feature = GeometryFeature(feature_tuples=stats.feature_tuples)
    features = [geometry_feature, ZscoreLayer(zscores), schnet_feature]
    combined_features = FeatureCombiner(
        features, distance_indices=distance_idx)

    # Next, we create the hidden architecture of CGnet
    arch = LinearLayer(feature_size, width)
    arch += LinearLayer(width, 1)

    # Finally, we assemble the model
    model = CGnet(arch, ForceLoss(), feature=combined_features,
                  priors=priors)
    return model, coords, embedding_property


def test_cgnet_mount():
    if not torch.cuda.is_available():
        raise SkipTest("GPU not available for testing.")
    device = torch.device('cuda')

    # This test asseses CUDA mounting for an entire CGnet model
    # First we create a random model with random protein data
    model, coords, embedding_property = generate_model()

    # Next, we mount the model to GPU
    model.mount(device)

    # Next, we check to see if each layer is mounted correctly
    # This is done by checking if parameters/buffers are mapped to the correct
    # device, or that feature classes are imbued with the appropriate device
    # First, we check features
    for layer in model.feature.layer_list:
        if isinstance(layer, (GeometryFeature, SchnetFeature)):
            assert layer.device == device
        if isinstance(layer, ZscoreLayer):
            assert layer.zscores.device.type == device.type
    # Next, we check priors
    for prior in model.priors:
        if isinstance(prior, HarmonicLayer):
            assert prior.harmonic_parameters.device.type == device.type
        if isinstance(prior, RepulsionLayer):
            assert prior.repulsion_parameters.device.type == device.type
    # Finally, we check the arch layers
    for param in model.parameters():
        assert param.device.type == device.type

    # Lastly, we perform a forward pass over the data and
    coords = torch.tensor(coords, requires_grad=True).to(device)
    embedding_property = embedding_property.to(device)
    pot, pred_force = model.forward(coords, embedding_property)
    assert pot.device.type == device.type
    assert pred_force.device.type == device.type


def test_cgnet_dismount():
    if not torch.cuda.is_available():
        raise SkipTest("GPU not available for testing.")
    device = torch.device('cuda')

    # This test asseses the ability of an entire CGnet to dismount from GPU
    # First we create a random model with random protein data
    model, coords, embedding_property = generate_model()

    # First, we mount the model to GPU
    model.mount(device)

    # Here we dismount the model from GPU
    device = torch.device('cpu')
    model.mount(device)

    # Next, we check to see if each layer is dismounted correctly
    # This is done by checking if parameters/buffers are mapped to the correct
    # device, or that feature classes are imbued with the appropriate device
    # First, we check features
    for layer in model.feature.layer_list:
        if isinstance(layer, (GeometryFeature, SchnetFeature)):
            assert layer.device.type == device.type
        if isinstance(layer, ZscoreLayer):
            assert layer.zscores.device.type == device.type
    # Next, we check priors
    for prior in model.priors:
        if isinstance(prior, HarmonicLayer):
            assert prior.harmonic_parameters.device.type == device.type
        if isinstance(prior, RepulsionLayer):
            assert prior.repulsion_parameters.device.type == device.type
    # Finally, we check the arch layers
    for param in model.parameters():
        assert param.device.type == device.type
    # Lastly, we perform a forward pass over the data and
    coords = torch.tensor(coords, requires_grad=True).to(device)
    pot, pred_force = model.forward(coords, embedding_property)
    assert pot.device.type == device.type
    assert pred_force.device.type == device.type

def test_save_load_model():
    # This test asseses the ability to dismount models from GPU that are loaded
    # from a saved .pt file
    if not torch.cuda.is_available():
        raise SkipTest("GPU not available for testing.")
    with tempfile.TemporaryDirectory() as tmp:
        device = torch.device('cuda')

        # This test asseses the ability of an entire CGnet to dismount from GPU
        # First we create a random model with random protein data
        model, coords, embedding_property = generate_model()

        # First, we mount the model to GPU
        model.mount(device)

        # Next we save the model to the temporary directory and load it again
        # to checkout if it can be dismounted from the GPU
        torch.save(model, tmp+"/cgnet_gpu_test.pt")
        del model
        loaded_model = torch.load(tmp+"/cgnet_gpu_test.pt")
        device = torch.device('cpu')
        loaded_model.mount(torch.device('cpu'))
        # First we check features 
        for layer in loaded_model.feature.layer_list:
            if isinstance(layer, (GeometryFeature, SchnetFeature)):
                assert layer.device.type == device.type
            if isinstance(layer, ZscoreLayer):
                assert layer.zscores.device.type == device.type
        # Next, we check priors
        for prior in loaded_model.priors:
            if isinstance(prior, HarmonicLayer):
                assert prior.harmonic_parameters.device.type == device.type
            if isinstance(prior, RepulsionLayer):
                assert prior.repulsion_parameters.device.type == device.type
        # Finally, we check the arch layers
        for param in loaded_model.parameters():
            assert param.device.type == device.type
        # Lastly, we perform a forward pass over the data and
        coords = torch.tensor(coords, requires_grad=True).to(device)
        pot, pred_force = loaded_model.forward(coords, embedding_property)
        assert pot.device.type == device.type
        assert pred_force.device.type == device.type
