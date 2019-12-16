# Author: Brooke Husic


import numpy as np

# These radii and masses were obtained from the following repository:
# https://github.com/ZiZ1/model_builder/blob/master/models/mappings/atom_types.py

# The radii were calculated using the molar volumes reported in:
# Haeckel, M., Hinz, H,-J., Hedwig, G. (1999). Partial molar volumes of
# proteins: amino acid side-chain contributions derived from the partial
# molar volumes of some tripeptides over the temperature range 10-90 C.
# Biophysical Chemistry. https://doi.org/10.1016/S0301-4622(99)00104-0

# radii are reported in NANOMETERS
RESIDUE_RADII = {
                'ALA': 0.1845, 'ARG': 0.3134,
                'ASN': 0.2478, 'ASP': 0.2335,
                'CYS': 0.2276, 'GLN': 0.2733,
                'GLU': 0.2639, 'GLY': 0.0000,
                'HIS': 0.2836, 'ILE': 0.2890,
                'LEU': 0.2887, 'LYS': 0.2938,
                'MET': 0.2916, 'PHE': 0.3140,
                'PRO': 0.2419, 'SER': 0.1936,
                'THR': 0.2376, 'TRP': 0.3422,
                'TYR': 0.3169, 'VAL': 0.2620
                }

# masses are reported in AMUS
RESIDUE_MASSES = {
                'ALA':   89.0935, 'ARG':  174.2017,
                'ASN':  132.1184, 'ASP':  133.1032,
                'CYS':  121.1590, 'GLN':  146.1451,
                'GLU':  147.1299, 'GLY':   75.0669,
                'HIS':  155.1552, 'ILE':  131.1736,
                'LEU':  131.1736, 'LYS':  146.1882,
                'MET':  149.2124, 'PHE':  165.1900,
                'PRO':  115.1310, 'SER':  105.0930,
                'THR':  119.1197, 'TRP':  204.2262,
                'TYR':  181.1894, 'VAL':  117.1469
                }


def calculate_bond_minima(bond_pairs, cgmolecule, units='Angstroms',
                          prefactor=0.7, names_to_include='all'):
    if units.lower() not in ['angstroms', 'nanometers']:
        raise ValueError("units must Angstroms or nanometers")

    resmap = cgmolecule.resmap
    resseq = cgmolecule.resseq
    if units == 'Angstroms':
        residue_radii = {k : 10*v for k, v in RESIDUE_RADII.items()}
    else:
        residue_radii = RESIDUE_RADII

    bond_minima = [(prefactor*residue_radii[resmap[resseq[b1]]] +
                    prefactor*residue_radii[resmap[resseq[b2]]])
                    for b1, b2 in bond_pairs]

    bond_minima = [np.round(bond, 4) for bond in bond_minima]

    return bond_minima







