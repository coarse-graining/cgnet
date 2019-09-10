# Author: Brooke Husic

import numpy as np
import torch

import mdtraj as md

from cgnet.feature import GeometryFeature
from cgnet.molecule import CGMolecule


# make a peptide backbone with 3 to 6 non-cap residues
# We need at least three for the angle test so we have
# three CA atoms.
residues = np.random.randint(3, 6)

# each non-cap residue will have 3 backbone atoms
names = ['C'] + ['N', 'CA', 'C'] * residues + ['N']
beads = len(names)

# resseq is the same length as names, where the integer corresponds
# to the residue assignment of that bead, so it looks like
# [1, 2, 2, 2, 3, 3, 3, ..., n] for n beads
resseq = [1] + list(np.concatenate([np.repeat(i+2, 3)
                                    for i in range(residues)])) + [2+residues]

# resmap maps the residue indices to amino acid identities; here we use
# all alanines - it doesn't matter
resmap = {1: 'ACE', (2+residues): 'NME'}
for r in range(residues):
    resmap[r+2] = 'ALA'

# we manually specify the bonds for some tests later
# in the format that mdtraj requires
bonds = np.zeros([beads-1, 4])
for b in range(beads-1):
    bonds[b] = [b, b+1, 0., 0.]

# create a pseudo-dataset with three dimensions
frames = np.random.randint(1, 10)
dims = 3

data = np.random.randn(frames, beads, dims)
data_tensor = torch.Tensor(data)


def test_cg_topology_standard():
    # Make sure topology works like an mdtraj topology (auto bond version)

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap,
                          bonds='standard')

    # Here we just make sure mdtraj.topology attributes have the right values
    assert molecule.top.n_atoms == beads
    assert molecule.top.n_bonds == beads-1 # 'standard' fills in the bonds
    assert molecule.top.n_chains == 1


def test_cg_topology_no_bonds():
    # Make sure topology works like an mdtraj topology (no bond version)

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap,
                          bonds=None)

    # Here we just make sure mdtraj.topology attributes have the right values
    assert molecule.top.n_atoms == beads
    assert molecule.top.n_bonds == 0 # no bonds!
    assert molecule.top.n_chains == 1


def test_cg_topology_custom_bonds():
    # Make sure topology works like an mdtraj topology (custom bond version)

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap,
                          bonds=bonds)

    assert molecule.top.n_atoms == beads
    assert molecule.top.n_bonds == beads-1 # manual number of bonds
    assert molecule.top.n_chains == 1


def test_cg_trajectory():
    # Make sure trajectory works like an mdtraj trajectory

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(data)

    # here we test that some mdtraj trajectory attibutes are right
    assert traj.n_frames == frames
    assert traj.top.n_atoms == beads
    assert traj.n_residues == residues + 2


def test_backbone_phi_dihedrals():
    # Make sure backbone phi dihedrals are correct

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(data)
    _, mdtraj_phis = md.compute_phi(traj)
    mdtraj_phis = np.abs(mdtraj_phis)

    # manual calculation of phi angles
    phis = []
    for frame_data in data:
        dihed_list = []
        for i in range(residues):
            # we get the phi's by starting at the 'N', which is the first
            # bead for every residue
            a = frame_data[i*3]
            b = frame_data[i*3+1]
            c = frame_data[i*3+2]
            # the last bead in the phi dihedral is the 'N' of the next residue
            d = frame_data[i*3+3]

            ba = b-a
            cb = c-b
            dc = d-c

            c1 = np.cross(ba, cb)
            c2 = np.cross(cb, dc)
            temp = np.cross(c2, c1)
            term1 = np.dot(temp, cb)/np.sqrt(np.dot(cb, cb))
            term2 = np.dot(c2, c1)
            dihed_list.append(np.arctan2(term1, term2))
        phis.append(dihed_list)

    phis = np.abs(phis)

    np.testing.assert_allclose(mdtraj_phis, phis, rtol=1e-4)


def test_backbone_psi_dihedrals():
    # Make sure backbone psi dihedrals are correct

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(data)
    _, mdtraj_psis = md.compute_psi(traj)
    mdtraj_psis = np.abs(mdtraj_psis)

    # manual calculation of psi angles
    psis = []
    for frame_data in data:
        dihed_list = []
        for i in range(residues):
            # we get the psi's by starting at the 'CA', which is the second
            # bead for every residue
            a = frame_data[i*3+1]
            b = frame_data[i*3+2]
            # the last two beads in the psi dihedral are the 'N' and 'CA'
            # of the next residue
            c = frame_data[i*3+3]
            d = frame_data[i*3+4]

            ba = b-a
            cb = c-b
            dc = d-c

            c1 = np.cross(ba, cb)
            c2 = np.cross(cb, dc)
            temp = np.cross(c2, c1)
            term1 = np.dot(temp, cb)/np.sqrt(np.dot(cb, cb))
            term2 = np.dot(c2, c1)
            dihed_list.append(np.arctan2(term1, term2))
        psis.append(dihed_list)

    psis = np.abs(psis)

    np.testing.assert_allclose(mdtraj_psis, psis, rtol=1e-4)


def test_equality_with_cgnet_dihedrals():
    # Make sure dihedrals are consistent with GeometryFeature

    geom_feature = GeometryFeature(n_beads=beads)
    out = geom_feature.forward(data_tensor)

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(data)

    mdtraj_phis = md.compute_phi(traj)[1]
    mdtraj_psis = md.compute_psi(traj)[1]

    mdtraj_phi_cosines = np.cos(mdtraj_phis)
    mdtraj_phi_sines = np.sin(mdtraj_phis)

    mdtraj_psi_cosines = np.cos(mdtraj_psis)
    mdtraj_psi_sines = np.sin(mdtraj_psis)

    # To get phi's and psi's out of cgnet, wen eed to specify which 
    # indices they correspond to along the backbone
    phi_inds = [i*3 for i in range(residues)] # ['N', 'CA', 'C', 'N'] dihedrals
    psi_inds = [i*3+1 for i in range(residues)] # ['CA', 'C', 'N', 'CA'] dihedrals

    cgnet_phi_cosines = geom_feature.dihedral_cosines.numpy()[:, phi_inds]
    cgnet_phi_sines = geom_feature.dihedral_sines.numpy()[:, phi_inds]

    cgnet_psi_cosines = geom_feature.dihedral_cosines.numpy()[:, psi_inds]
    cgnet_psi_sines = geom_feature.dihedral_sines.numpy()[:, psi_inds]

    np.testing.assert_allclose(mdtraj_phi_cosines, cgnet_phi_cosines,
                               rtol=1e-4)
    np.testing.assert_allclose(mdtraj_phi_sines, cgnet_phi_sines,
                               rtol=1e-4)
    np.testing.assert_allclose(mdtraj_psi_cosines, cgnet_psi_cosines,
                               rtol=1e-4)
    np.testing.assert_allclose(mdtraj_psi_sines, cgnet_psi_sines,
                               rtol=1e-4)


def test_equality_with_cgnet_distances():
    # Make sure CA distances are consistent with GeometryFeature

    geom_feature = GeometryFeature(n_beads=beads)
    out = geom_feature.forward(data_tensor)

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(data)

    # Calculate all pairs of CA distances
    CA_inds = [i for i, name in enumerate(names) if name == 'CA']
    CA_pairs = [] # these are feature tuples
    for i, ind1 in enumerate(CA_inds[:-1]):
        for j, ind2 in enumerate(CA_inds[i+1:]):
            CA_pairs.append((ind1, ind2))
    mdtraj_CA_dists = md.compute_distances(traj, CA_pairs)

    # map each CA distance feature tuple to  the integer index
    CA_feature_tuple_dict = {key: i for i, key in
                             enumerate(geom_feature.descriptions['Distances'])
                             if key in CA_pairs}

    # retrieve CA distances only from the feature object
    cgnet_CA_dists = geom_feature.distances.numpy()[:, [CA_feature_tuple_dict[key]
                                             for key in CA_pairs]]

    np.testing.assert_allclose(mdtraj_CA_dists, cgnet_CA_dists, rtol=1e-6)


def test_equality_with_cnget_angles():
    # Make sure CA distances caluclated internally are consistent with mdtraj.
    # This test appears here because it requires an mdtraj dependency.
    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(data)

    # Grab the CA inds only to get the backbone angles and compute them
    # with mdtraj
    CA_inds = [i for i, name in enumerate(names) if name == 'CA']
    backbone_angles = [(CA_inds[i], CA_inds[i+1], CA_inds[i+2])
                        for i in range(len(CA_inds)-2)]
    mdtraj_angles = md.compute_angles(traj, backbone_angles)

    # Get the GeometryFeature for just the 
    geom_feature = GeometryFeature(feature_tuples=backbone_angles)
    out = geom_feature.forward(data_tensor)

    cgnet_angles = geom_feature.angles

    np.testing.assert_allclose(mdtraj_angles, cgnet_angles, rtol=1e-5)
