# Author: Brooke Husic

import numpy as np
import torch

import mdtraj as md

from cgnet.feature import ProteinBackboneFeature
from cgnet.molecule import CGMolecule


# make a peptide backbone with 1-5 non-cap residues
residues = np.random.randint(2, 6)
names = ['C'] + ['N', 'CA', 'C'] * residues + ['N']
beads = len(names)

resseq = [1] + list(np.concatenate([np.repeat(i+2, 3)
									for i in range(residues)])) + [2+residues]
resmap = {1 : 'ACE', (2+residues): 'NME'}
for r in range(residues):
	resmap[r+2] = 'ALA'

bonds = np.zeros([beads-1, 4])
for b in range(beads-1):
	bonds[b] = [b, b+1, 0., 0.]


frames = np.random.randint(1, 10)
dims = 3

x = np.random.randn(frames, beads, dims)
xt = torch.Tensor(x)


def test_cg_topology_standard():
    # Make sure topology works like an mdtraj topology

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap,
    					  bonds='standard')

    assert molecule.top.n_atoms == beads
    assert molecule.top.n_bonds == beads-1
    assert molecule.top.n_chains == 1


def test_cg_topology_no_bonds():
    # Make sure topology works like an mdtraj topology

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap,
    					  bonds=None)

    assert molecule.top.n_atoms == beads
    assert molecule.top.n_bonds == 0
    assert molecule.top.n_chains == 1


def test_cg_topology_custom_bonds():
    # Make sure topology works like an mdtraj topology

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap,
    					  bonds=bonds)

    assert molecule.top.n_atoms == beads
    assert molecule.top.n_bonds == beads-1
    assert molecule.top.n_chains == 1


def test_cg_trajectory():
    # Make sure trajectory works like an mdtraj trajectory

    molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(x)

    assert traj.n_frames == frames
    assert traj.top.n_atoms == beads
    assert traj.n_residues == residues + 2


def test_backbone_phi_dihedrals():
	# Make sure backbone phi dihedrals are correct

	molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
	traj = molecule.make_trajectory(x)
	_, mdtraj_phis = md.compute_phi(traj)
	mdtraj_phis = np.abs(mdtraj_phis)

	phis = []
	for j, z in enumerate(x):
	    dihed_list = []
	    for i in range(residues):
	        a = z[i*3]
	        b = z[i*3+1]
	        c = z[i*3+2]
	        d = z[i*3+3]
	        
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
	traj = molecule.make_trajectory(x)
	_, mdtraj_psis = md.compute_psi(traj)
	mdtraj_psis = np.abs(mdtraj_psis)

	psis = []
	for j, z in enumerate(x):
	    dihed_list = []
	    for i in range(residues):
	        a = z[i*3+1]
	        b = z[i*3+2]
	        c = z[i*3+3]
	        d = z[i*3+4]
	        
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
	# Make sure dihedrals are consistent with ProteinBackboneFeature

	f = ProteinBackboneFeature()
	out = f.forward(xt)

	molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
	traj = molecule.make_trajectory(x)

	mdtraj_phis = md.compute_phi(traj)[1]
	mdtraj_psis = md.compute_psi(traj)[1]

	mdtraj_phi_cosines = np.cos(mdtraj_phis)
	mdtraj_phi_sines = np.sin(mdtraj_phis)

	mdtraj_psi_cosines = np.cos(mdtraj_psis)
	mdtraj_psi_sines = np.sin(mdtraj_psis)

	phi_inds = [i*3 for i in range(residues)]
	psi_inds = [i*3+1 for i in range(residues)]

	cgnet_phi_cosines = f.dihedral_cosines.numpy().T[phi_inds].T
	cgnet_phi_sines = f.dihedral_sines.numpy().T[phi_inds].T

	cgnet_psi_cosines = f.dihedral_cosines.numpy().T[psi_inds].T
	cgnet_psi_sines = f.dihedral_sines.numpy().T[psi_inds].T

	np.testing.assert_allclose(mdtraj_phi_cosines, cgnet_phi_cosines,
							   rtol=1e-5)
	np.testing.assert_allclose(mdtraj_phi_sines, cgnet_phi_sines,
							   rtol=1e-5)
	np.testing.assert_allclose(mdtraj_psi_cosines, cgnet_psi_cosines,
							   rtol=1e-5)
	np.testing.assert_allclose(mdtraj_psi_sines, cgnet_psi_sines,
							   rtol=1e-5)


def test_equality_with_cgnet_distances():
	# Make sure CA distances are consistent with ProteinBackboneFeature

	f = ProteinBackboneFeature()
	out = f.forward(xt)

	molecule = CGMolecule(names=names, resseq=resseq, resmap=resmap)
	traj = molecule.make_trajectory(x)

	CA_inds = [i for i, name in enumerate(names) if name == 'CA']
	CA_pairs = []
	for i, ind1 in enumerate(CA_inds[:-1]):
	    for j, ind2 in enumerate(CA_inds[i+1:]):
	        CA_pairs.append((ind1, ind2))
	mdtraj_CA_dists = md.compute_distances(traj, CA_pairs)

	CA_ind_dict = {key:i for i, key in enumerate(f.descriptions['Distances'])
 						 if key in CA_pairs}
	cgnet_CA_dists = f.distances.numpy()[:, [CA_ind_dict[key]
 									         for key in CA_pairs]]

	np.testing.assert_allclose(mdtraj_CA_dists, cgnet_CA_dists)
