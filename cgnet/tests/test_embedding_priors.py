# Author: Nick Charron 

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryStatistics, EmbeddingStatistics, GeometryFeature
from cgnet.network import (EmbeddingHarmonicLayer, EmbeddingRepulsionLayer,
                           PriorForceComputer)

# The following sets up our pseud-simulation data

# Number of frames
frames = np.random.randint(5, 10) * 2 #double this to avoid nonzero variances

# Number of coarse-grained beads. We need at least 8 so we can do
# dihedrals in the backbone tests (where every other atom is designated
# as a backbone atom)
beads = np.random.randint(8, 20)

# Number of dimensions; for now geometry only handles 3
dims = 3

# Create a pseudo simulation dataset, but double each frame to get nonzero variances
data = np.random.randn(frames, beads, dims)

# Create embeddings that vary from frame to frame to simulate multiple
# different molecules within the same dataset

embeddings = np.random.randint(low=1, high=10, size=(int(frames/2),beads)) #again,double this so
# we dont get nonzero variances.

embeddings = np.tile(embeddings, (2,1))

# random temperature
temperature = np.random.uniform(low=250,high=350)
KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184
beta = JPERKCAL/KBOLTZMANN/AVOGADRO/temperature

stats = EmbeddingStatistics(data, embeddings, backbone_inds='all',
                           get_all_distances=True,
                           get_backbone_angles=True,
                           get_backbone_dihedrals=True,
                           get_redundant_distance_mapping=True,
                           temperature=temperature)
beta = stats.beta


def test_embedding_harmonic_layer():
    # Tests to make sure that interaction parameters are assembled 
    # properly according to the input embeddings, resulting in the
    # correct energy predictions for harmonic interactions

    # First, we construct a random set of distances and angles
    # with which we can constrain with an embedding dependent prior

    num_bonds = np.random.randint(1, high=len(stats._distance_pairs))
    num_angles = np.random.randint(1, high=len(stats._angle_trips))

    bond_idx = np.random.choice(np.arange(0, len(stats._distance_pairs)),
                                num_bonds, replace=False)
    angle_idx = np.random.choice(np.arange(0, len(stats._angle_trips)),
                                num_angles, replace=False)

    bond_features = [stats._distance_pairs[i] for i in bond_idx]
    angle_features = [stats._angle_trips[i] for i in angle_idx]
    #print(bond_features)
    bond_dict = stats.get_prior_statistics(bond_features)
    angle_dict = stats.get_prior_statistics(angle_features)
    #print(bond_dict)
    # Here, the callback indices are just placeholders, though
    # we still calculate them in the intended fashion using the 
    # stats.return_indices method

    bond_callback_idx = stats.return_indices(bond_features)
    angle_callback_idx = stats.return_indices(angle_features)

    # Next, we construct embedding harmonic layers for both feature sets

    bond_embedding_hlayer = EmbeddingHarmonicLayer(bond_callback_idx,
                                bond_dict["Distances"], bond_features)
    angle_embedding_hlayer = EmbeddingHarmonicLayer(angle_callback_idx,
                                angle_dict["Angles"], angle_features)

    #print(bond_embedding_hlayer.parameter_dict)

    # we produce a manual calculation of the energies for each
    # feature set

    distances = torch.tensor(stats.distances[:, bond_idx])
    angles = torch.tensor(stats.angles[:, angle_idx])
    embeddings = torch.tensor(stats.embeddings)

    # first, we handle the bonds
    bond_tuples = torch.tensor(bond_features)
    embedding_tuples = embeddings[:, bond_tuples]
    num_examples = distances.size()[0]
    bond_tensor_lookups = torch.cat((bond_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    bond_means = torch.tensor([[bond_dict["Distances"][tuple(lookup)]['mean']
                          for lookup in bond_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    bond_constants = torch.tensor([[bond_dict["Distances"][tuple(lookup)]['k']
                          for lookup in bond_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_bond_energy = torch.sum(bond_constants * ((distances - bond_means)**2),
                             dim=1).reshape(num_examples, 1) / 2.0

    # next, we handle the angles
    angle_tuples = torch.tensor(angle_features)
    embedding_tuples = embeddings[:, angle_tuples]
    num_examples = angles.size()[0]
    angle_tensor_lookups = torch.cat((angle_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    angle_means = torch.tensor([[angle_dict["Angles"][tuple(lookup)]['mean']
                          for lookup in angle_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    angle_constants = torch.tensor([[angle_dict["Angles"][tuple(lookup)]['k']
                          for lookup in angle_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_angle_energy = torch.sum(angle_constants * ((angles - angle_means)**2),
                              dim=1).reshape(num_examples, 1) / 2.0

    # Lastly, we compute the same energies using the EmbeddingHarmonicLayers
    # and compare their output with the manual calculations above

    bond_energy = bond_embedding_hlayer(distances, embeddings)
    angle_energy = angle_embedding_hlayer(angles, embeddings)

    np.testing.assert_equal(manual_bond_energy.numpy(),
                            bond_energy.numpy())
    np.testing.assert_equal(manual_angle_energy.numpy(),
                            angle_energy.numpy())


def test_embedding_repulsion_layer():
    # Tests to make sure that interaction parameters are assembled 
    # properly according to the input embeddings, resulting in the
    # correct energy predictions for repulsion interactions

    # First, we construct a random set of distances and angles
    # with which we can constrain with an embedding dependent prior

    num_distances = np.random.randint(1, high=len(stats._distance_pairs))

    distance_idx = np.random.choice(np.arange(0, len(stats._distance_pairs)),
                                num_distances, replace=False)

    features = [stats._distance_pairs[i] for i in distance_idx]
    feature_dict = stats.get_prior_statistics(features)

    # Here, we construct a parameter dictionary of random exponents
    # and excluded volumes

    parameter_dictionary = {}
    for key in feature_dict["Distances"].keys():
        parameter_dictionary[key] = {}
        parameter_dictionary[key]["ex_vol"] = np.random.uniform(1.0, high=5.0)
        parameter_dictionary[key]["exp"] = np.random.uniform(1.0, high=5.0)

    # Here, the callback indices are just placeholders, though
    # we still calculate them in the intended fashion using the 
    # stats.return_indices method

    callback_idx = stats.return_indices(features)

    # Next, we construct embedding harmonic layers for both feature sets

    embedding_rlayer = EmbeddingRepulsionLayer(callback_idx,
                                parameter_dictionary, features)

    #print(bond_embedding_hlayer.parameter_dict)

    # we produce a manual calculation of the energies for each
    # feature set

    distances = torch.tensor(stats.distances[:, distance_idx])
    embeddings = torch.tensor(stats.embeddings)

    # first, we handle the bonds
    feature_tuples = torch.tensor(features)
    embedding_tuples = embeddings[:, feature_tuples]
    num_examples = distances.size()[0]
    tensor_lookups = torch.cat((feature_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    ex_vols = torch.tensor([[parameter_dictionary[tuple(lookup)]['ex_vol']
                          for lookup in tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    exps = torch.tensor([[parameter_dictionary[tuple(lookup)]['exp']
                          for lookup in tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_repulsion_energy = torch.sum((ex_vols / distances) ** exps,
                                  dim=1).reshape(num_examples, 1) / 2.0

    # Lastly, we compute the same energies using the EmbeddingHarmonicLayers
    # and compare their output with the manual calculations above

    repulsion_energy = embedding_rlayer(distances, embeddings)

    np.testing.assert_equal(manual_repulsion_energy.numpy(),
                            repulsion_energy.numpy())


def test_prior_computer_with_embeddings():
    # Tests to make sure that the correct prior forces are calculated
    # set up embedding harmonic layer and embedding repulsion layer as prior list
    # We set all distances to have repulsive interactions, and a subset of these 
    # distances will also have harmonic interactions

    # random subest for bonds:
    upper_idx = np.random.randint(4,high=10)
    bonds_idx = np.arange(upper_idx).astype('int32')

    # here we assemble the harmonix parameter dictionary
    distances_interactions = stats.get_prior_statistics(stats.descriptions['Distances'])['Distances']
    distances_idx = stats.return_indices(stats.descriptions['Distances'])
    bond_bead_tuples = [stats.descriptions['Distances'][i] for i in bonds_idx]
    bond_bead_embedding_keys = [key for key in distances_interactions.keys() if
                                (key[0], key[1]) in bond_bead_tuples]
    bond_interactions = {key:value for key, value in distances_interactions.items() if
                         key in bond_bead_embedding_keys}

    # Here we assemble the repulsive paramter dictionary
    # Instead of using that stats object, we just get some random ex_vols and
    # exps for each embedding-dependent tuple:

    repul_interactions = {}
    for key in distances_interactions.keys():
        repul_interactions[key] = {}
        repul_interactions[key]['ex_vol'] = torch.tensor(np.random.uniform(1,5))
        repul_interactions[key]['exp'] = torch.tensor(np.random.uniform(1,5))

    # Here, we instance the embedding-dependent priors
    embedding_hlayer = EmbeddingHarmonicLayer(bonds_idx,
                                bond_interactions, bond_bead_tuples)
    embedding_rlayer = EmbeddingRepulsionLayer(distances_idx,
                                repul_interactions, stats.descriptions['Distances'])

    geom_feat = GeometryFeature(feature_tuples='all_backbone',
                                n_beads=beads)

    # Produce the tensor of invariant features from the coordinates
    coords = torch.tensor(data, requires_grad=True)
    geom_feat_out = geom_feat(coords)
    # Here, we manually compute the total prior energy/forces

    bonds = geom_feat_out[:, bonds_idx]
    distances = geom_feat_out[:, distances_idx]
    embeddings = torch.tensor(stats.embeddings)

    # first, we handle the bonds
    bond_tuples = torch.tensor(bond_bead_tuples)
    embedding_tuples = embeddings[:, bond_tuples]
    num_examples = distances.size()[0]
    bond_tensor_lookups = torch.cat((bond_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    bond_means = torch.tensor([[bond_interactions[tuple(lookup)]['mean']
                          for lookup in bond_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    bond_constants = torch.tensor([[bond_interactions[tuple(lookup)]['k']
                          for lookup in bond_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_bond_energy = torch.sum(bond_constants * ((bonds - bond_means)**2),
                             dim=1).reshape(num_examples, 1) / 2.0

    # next, we handle the repulsions
    distances_tuples = torch.tensor(stats.descriptions['Distances'])
    embedding_tuples = embeddings[:, distances_tuples]
    distances_tensor_lookups = torch.cat((distances_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    distances_ex_vols = torch.tensor([[repul_interactions[tuple(lookup)]['ex_vol']
                          for lookup in distances_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    distances_exps = torch.tensor([[repul_interactions[tuple(lookup)]['exp']
                          for lookup in distances_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_repulsion_energy = torch.sum((distances_ex_vols / distances) ** distances_exps,
                              dim=1).reshape(num_examples, 1) / 2.0

    total_manual_energy = torch.sum(manual_bond_energy + manual_repulsion_energy)
    total_manual_forces = torch.autograd.grad(-total_manual_energy, coords,
        create_graph=True, retain_graph=True)[0]

    # Next, we create a PriorForceComputer module
    prior_computer = PriorForceComputer([embedding_hlayer, embedding_rlayer],
                         geom_feat)

    prior_forces = prior_computer(coords, embeddings)
    np.testing.assert_equal(total_manual_forces.detach().numpy(), prior_forces.detach().numpy())



def test_embedding_prior_pipline():
    # This test ensures that embedding-based priors can be successfully
    # incoroprated into the full CGSchNet pipeline
    pass
