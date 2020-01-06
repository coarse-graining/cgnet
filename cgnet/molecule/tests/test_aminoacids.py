# Author: Brooke Husic

import numpy as np
import itertools

from cgnet.molecule import (CGMolecule, RESIDUE_RADII,
                            calculate_hard_sphere_minima)

def test_angstrom_conversion():

    # Create a CG molecule with each amino acid doubled
    all_residues = list(RESIDUE_RADII.keys())
    doubled_res_list = np.concatenate(np.vstack([all_residues,
                                                 all_residues]).T)
    names = ['CA'] * len(doubled_res_list)
    resseq = np.arange(1, len(doubled_res_list)+1)
    resmap = {i+1 : doubled_res_list[i] for i in range(len(doubled_res_list))}
    mypeptide = CGMolecule(names, resseq, resmap)

    # Enumerate the bonds between same residues only
    same_res_bonds = [(i, i+1) for i in range(len(doubled_res_list))
                      if i % 2 == 0]

    # Calculate the minima with a prefactor of 1.0
    same_res_minima = calculate_hard_sphere_minima(same_res_bonds,
                                                   mypeptide, prefactor=1.0)

    # The minima should be the radii in nanometers after a factor of 1/20
    single_nm_radii = [i/20 for i in same_res_minima]

    values_from_dict = [RESIDUE_RADII[res] for i, res in
                        enumerate(doubled_res_list) if i % 2 == 0]

    np.testing.assert_allclose(values_from_dict, single_nm_radii)

# def test_minima_calculations():

def test_CA_vs_CB_minima_correspondence():
    # This tests that CA-CA distances are the same as CB-CB for the same
    # residue pair

    # Shuffle the twenty amino acids to make a peptide from
    possible_residues = list(RESIDUE_RADII.keys())
    np.random.shuffle(possible_residues)

    # Make a CA+CB CGMolecule object with a random number of residues
    num_residues = np.random.randint(3, 10)
    names = ['CA', 'CB'] * num_residues
    resseq = list(np.concatenate([np.repeat(i+1, 2) for i in range(num_residues)]))
    resmap = {i+1 : possible_residues[i] for i in range(num_residues)}
    mypeptide = CGMolecule(names, resseq, resmap)

    # Enumerate each set of inds
    CA_inds = [i for i in range(num_residues*2) if i % 2 == 0]
    CB_inds = [i for i in range(num_residues*2) if i % 2 == 1]

    # Enumerate each set of bonds
    CA_CA_bonds = list(itertools.combinations(CA_inds, 2))
    CB_CB_bonds = list(itertools.combinations(CB_inds, 2))

    # Calculate each set of minima
    CA_CA_minima = calculate_hard_sphere_minima(CA_CA_bonds, mypeptide)
    CB_CB_minima = calculate_hard_sphere_minima(CB_CB_bonds, mypeptide)

    # Ensure equality
    np.testing.assert_array_equal(CA_CA_minima, CB_CB_minima)
