# Author: B Husic


import numpy as np
import mdtraj as md


class CGMolecule():
    """Casting of a coarse-grained (CG) molecule as an mdtraj-compatible
    topology with the option to input trajectory coordinates to create
    an mdtraj Trajectory object that can be used for standard analyses
    such as the computation of dihedral angles, contact distances, etc.

    Please refer to the mdtraj documentation at http://www.mdtraj.org
    or the code at https://github.com/mdtraj/mdtraj.

    Attributes
    ----------
    names : list
        List of atom names in the CG molecule
    resseq : list
        List of residue assignments of each atom in the CG molecule
    resmap: dictionary
        List of residue indexes (key) and corresponding residue name (value)
    elements : list or None
        List of elements corresponding to each atom in names. If None,
        then the first character of the corresponding name string is used.
    bonds : np.array, 'standard', or None (default='standard')
        If None, no bonds. If 'standard', applies mdtraj.create_standard_bonds
        after topology is constructed. If np.array, bonds are given manually
        with a np.array of dimensions (n_bonds, 4) with zeroes everywhere
        except the ZERO-INDEXED indices of the bonded atoms
    starting_index : int (default=0)
        Index for first atom in CG molecule if something other than a 0-index
        is desired

    Example
    -------
    # Alanine dipeptide backbone
    names = ['C', 'N', 'CA', 'C', 'N']
    resseq = [1, 2, 2, 2, 3]
    resmap = {1 : 'ACE', 2 : 'ALA', 3 : 'NME'}
    molecule  = CGMolecule(names=names, resseq=resseq, resmap=resmap)
    traj = molecule.make_trajectory(coordinates)

    Notes
    -----
    Currently there is no option to have more than one chain
    Unitcells are not implemented

    References
    ----------
    McGibbon, R. T., Beauchamp, K. A., Harrigan, M. P., Klein, C.,
        Swails, J. M., Hernández, C. X., Schwantes, C. R., Wang, L.-P.,
        Lane, T. J., and Pande, V. S. (2015). MDTraj: A Modern Open Library
        for the Analysis of Molecular Dynamics Trajectories. Biophys J.
        http://dx.doi.org/10.1016/j.bpj.2015.08.015
    """

    def __init__(self, names, resseq, resmap, elements=None,
                 bonds='standard', starting_index=0):
        if len(names) != len(resseq):
            raise ValueError(
                'Names and resseq must be lists of the same length')
        self.names = names
        self.resseq = resseq

        if elements is None:
            # this may not be a good idea
            elements = [name[0] for name in self.names]
        self.elements = elements

        if not np.array_equal(sorted(resmap.keys()), np.unique(resseq)):
            raise ValueError('resmap dictionary must have a key for \
                              each index in resseq')
        self.resmap = resmap
        self.bonds = bonds
        self.starting_index = starting_index

        self.make_topology()

    def make_topology(self):
        """Generates an mdtraj.Topology object."""
        pd = md.utils.import_('pandas')
        data = []
        for i, name in enumerate(self.names):
            row = (i + self.starting_index, name, name[0], self.resseq[i],
                   self.resmap[self.resseq[i]], 0, '')
            data.append(row)
        atoms = pd.DataFrame(data,
                            columns=["serial", "name", "element", "resSeq",
                                     "resName", "chainID", "segmentID"])
        if self.bonds == 'standard':
            top = md.Topology.from_dataframe(atoms, None)
            top.create_standard_bonds()
        else:
            top = md.Topology.from_dataframe(atoms, self.bonds)

        self.top = top
        self.topology = top

    def make_trajectory(self, coordinates):
        """Generates an mdtraj.Trajectory object."""
        if len(coordinates.shape) != 3:
            raise ValueError('coordinates must be a np.array of shape \
                              [frames, atoms, dimensions]')
        if coordinates.shape[1] != self.top.n_atoms:
            raise ValueError('coordinates dimension 1 must be the number \
                              of atoms')
        if coordinates.shape[2] != 3:
            raise ValueError('coordinates must have 3 dimensions')

        # this is a hack; NOT recommended for actual use of mdtraj
        return md.core.trajectory.Trajectory(coordinates, self.top)
