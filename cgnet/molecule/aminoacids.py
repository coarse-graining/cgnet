# Author: Brooke Husic


import numpy as np
import warnings

# These radii and masses were obtained from the following repository:
# https://github.com/ZiZ1/model_builder/blob/master/models/mappings/atom_types.py

# The radii were calculated by assuming a sphere and solving for the radius
# using the molar volumes at 25 Celcius reported in Table 6, column 1, of:
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


def calculate_hard_sphere_minima(bead_pairs, cgmolecule, units='Angstroms',
                                 prefactor=0.7):
    """This function uses amino acid radii to calculate a minimum contact
    distance between atoms in a CGMolecule in either Angstroms or nanometers.
    Both glycine-glycine pairs and atoms in the same residue will return
    a distance of zero (the latter will also raise a warning). See also Notes,
    below.

    Parameters
    ----------
    bead_pairs : list of two-element tuples
        Each tuple contains the two atom indices in the coarse-grained for
        which a mininum distance should be calculated.
    cgmolecule : cgnet.molecule.CGMolecule instance
        An initialized CGMolecule object.
    units : 'Angstroms' or 'nanometers' (default='Angstroms')
        The unit in which the minimum distances should be returned
    prefactor : float (default=0.7)
        Factor by which each atomic radii should be multiplied.
        The default of 0.7 is inspired by reference [1].

    Returns
    -------
    hard_sphere_minima : list of floats
        Each element contains the minimum hard sphere distance corresponding
        to the same index in the input list of bead_pairs

    Notes
    -----
    This method does NOT take into account the identity of the atom in the 
    residue. In other words, the CA-CA, CA-CB, CB-CB, etc. distances will all
    be identical between two residues. In the example provided below, the
    hard_sphere_minima output will be a list of two identical distances.

    This method does NOT account for distances between atoms within the same
    residue (i.e., the same residue index). These distances will be returned
    as zero and a warning will be raised. If you are using this method to 
    populate a repulsion prior, consider applying harmonic priors to such
    intra-residue distances.

    References
    ----------
    [1] Cheung, M. S., Finke, J. M., Callahan, B., Onuchic, J. N. (2003).
        Exploring the interplay between topology and secondary structure
        formation in the protein folding problem. J. Phys. Chem. B.
        https://doi.org/10.1021/jp034441r

    Example
    -------
    names = ['CA', 'CB', 'CA', 'CB']
    resseq = [1, 1, 2, 2]
    resmap = {1 : 'ALA', 2 : 'PHE'}

    dipeptide = CGMolecule(names, resseq, resmap)

    # Our CA-CA distance is (0, 2), and our CB-CB distance is (1, 3)
    hard_sphere_minima = calculate_hard_sphere_minima([(0, 2), (1, 3)],
                                                       dipeptide)

    # Note that in this example, hard_sphere_minima will have two entries
    # with the same distance
    """
    if units.lower() not in ['angstroms', 'nanometers']:
        raise ValueError("units must Angstroms or nanometers")

    resmap = cgmolecule.resmap
    resseq = cgmolecule.resseq
    if units == 'Angstroms':
        residue_radii = {k : 10*v for k, v in RESIDUE_RADII.items()}
    else:
        residue_radii = RESIDUE_RADII

    # Calculate the distance unless the residue indices are the same,
    # in which case use a nan instead. We go through nans because we
    # want to provide the user with the problematic indices, and zeros
    # aren't unique because a GLY-GLY pair would also return a zero
    # even for different residue indices.
    hard_sphere_minima = np.array(
                    [(prefactor*residue_radii[resmap[resseq[b1]]] +
                    prefactor*residue_radii[resmap[resseq[b2]]])
                    if resseq[b1] != resseq[b2] else np.nan
                    for b1, b2 in bead_pairs]
                    )

    nan_indices = np.where(np.isnan(hard_sphere_minima))[0]
    if len(nan_indices) > 0:
        warnings.warn("The following bead pairs were in the same residue. Their "
                      "minima were set to zero: {}".format(
                      [bead_pairs[ni] for ni in nan_indices]))
        hard_sphere_minima[nan_indices] = 0.

    hard_sphere_minima = [np.round(dist, 4) for dist in hard_sphere_minima]

    return hard_sphere_minima
