# Author: Brooke Husic

import numpy as np
import itertools

from cgnet.molecule import (CGMolecule, RESIDUE_RADII,
                            calculate_hard_sphere_minima)

# Shuffle the twenty amino acids to make a peptide from
possible_residues = list(RESIDUE_RADII.keys())
np.random.shuffle(possible_residues)

# Make a CA+CB CGMolecule object with a random number of residues
num_residues = np.random.randint(3, 10)

names = ['CA', 'CB'] * num_residues
resseq = list(np.concatenate([np.repeat(i+1, 2) for i in range(num_residues)]))
resmap = {i+1 : possible_residues[i] for i in range(num_residues)}

mypeptide = CGMolecule(names, resseq, resmap)

def test_angstrom_conversion():
    

def test_minima_calculations():

def test_CA_CB_correspondence():
    # This tests that CA-CA distances are the same as CB-CB for the same
    # residue pair

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
