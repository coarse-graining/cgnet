# Author: B Husic
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial


KBOLTZMANN = 1.38064852e-23
AVOGARDO = 6.022140857e23
JPERKCAL = 4184


class ProteinBackboneStatistics():
    """Calculation of statistics for protein backbone features; namely
   distances, angles, and dihedral cosines and sines.

    Parameters
    ----------
    data : torch.Tensor or np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    get_distances : Boolean (default=True)
        Whether to calculate distances
    get_angles : Boolean (default=True)
        Whether to calculate angles
    get_dihedrals : Boolean, (default=True)
        Whether to calculate dihedral cosines and sines
    temperature : float (default=300.0)
        Temperature of system

    Attributes
    ----------
    stats_dict : dictionary
        Stores 'mean' and 'std' for caluclated features
    descriptions : dictionary
        List of indices (value) for each feature type (key)

    Example
    -------
    ds = ProteinBackboneStatistics(data, n_beads = 10)
    print(ds.stats_dict['Distances']['mean'])
    """

    def __init__(self, data,
                 get_distances=True, get_angles=True,
                 get_dihedrals=True, temperature=300.0):
        if torch.is_tensor(data):
            self.data = data.numpy()
        else:
            self.data = data
        self.n_frames = self.data.shape[0]
        self.n_beads = self.data.shape[1]
        self.temperature = temperature

        self._get_distance_indices()
        self.stats_dict = {}

        self.distances = None
        self.angles = None
        self.dihedral_cosines = None
        self.dihedral_sines = None

        self._name_dict = {
            'Distances': self.distances,
            'Angles': self.angles,
            'Dihedral_cosines': self.dihedral_cosines,
            'Dihedral_sines': self.dihedral_sines
        }
        self.descriptions = {}

        if get_distances:
            self._get_pairwise_distances()
            self._name_dict['Distances'] = self.distances
            self._get_stats(self.distances, 'Distances')

        if get_angles:
            self._get_angles()
            self._name_dict['Angles'] = self.angles
            self._get_stats(self.angles, 'Angles')

        if get_dihedrals:
            self._get_dihedrals()
            self._name_dict['Dihedral_cosines'] = self.dihedral_cosines
            self._name_dict['Dihedral_sines'] = self.dihedral_sines
            self._get_stats(self.dihedral_cosines, 'Dihedral_cosines')
            self._get_stats(self.dihedral_sines, 'Dihedral_sines')

    def _get_key(self, key, name):
        if name == 'Dihedral_cosines':
            return tuple(list(key) + ['cos'])
        if name == 'Dihedral_sines':
            return tuple(list(key) + ['sin'])
        else:
            return key

    def _flip_dict(self, mydict):
        all_inds = np.unique(np.sum([list(mydict[stat].keys())
                                     for stat in mydict.keys()]))

        newdict = {}
        for i in all_inds:
            newdict[i] = {}
            for stat in mydict.keys():
                if i in mydict[stat].keys():
                    newdict[i][stat] = mydict[stat][i]
        return newdict

    def get_zscores(self, tensor=True, as_dict=True, flip_dict=False,
                    order=["Distances", "Angles",
                           "Dihedral_cosines", "Dihedral_sines"]):
        """Obtain zscores (mean and standard deviation) for features

        Parameters
        ----------
        tensor : Boolean (default=True)
            Returns type torch.Tensor if True and np.array if False
        order : List, default=['Distances', 'Angles', 'Dihedral_cosines',
                               'Dihedral_sines']
            Order of statistics in output

        Returns
        -------
        zscore_array : torch.Tensor or np.array
            2 by n tensor/array with means in the first row and
            standard deviations in the second row, where n is
            the number of features
        """
        for key in order:
            if self._name_dict[key] is None:
                raise ValueError("{} have not been calculated".format(key))

        zscore_keys = np.sum([[self._get_key(key, name)
                               for key in self.descriptions[name]]
                              for name in order])
        zscore_array = np.vstack([
            np.concatenate([self.stats_dict[key][stat]
                            for key in order]) for stat in ['mean', 'std']])
        if not tensor:
            zscore_array = torch.from_numpy(zscore_array)

        if as_dict:
            zscore_dict = {}
            for i, stat in enumerate(['mean', 'std']):
                zscore_dict[stat] = dict(zip(zscore_keys, zscore_array[i, :]))
            if flip_dict:
                zscore_dict = self._flip_dict(zscore_dict)
            return zscore_dict
        else:
            return zscore_array

    def get_bond_constants(self, tensor=True, as_dict=True, zscores=True,
                           flip_dict=False,
                           order=["Distances", "Angles",
                                  "Dihedral_cosines", "Dihedral_sines"]):
        """Obtain bond constants (K values and means) for adjacent distance
           and angle features. K values depend on the temperature.

        Parameters
        ----------
        tensor : Boolean (default=True)
            Returns type torch.Tensor if True and np.array if False

        Returns
        -------
        bondconst_array : torch.Tensor or np.array
            2 by n tensor/array with means in the first row and
            standard deviations in the second row, where n is
            the number of adjacent pairwise distances plus the
            number of angles
        """
        if zscores and not as_dict:
            raise RuntimeError('zscores can only be True if as_dict is True')

        if self.distances is None or self.angles is None:
            raise RuntimeError('Must compute distances and angles in \
                                order to get bond constants')

        self.kb = JPERKCAL/KBOLTZMANN/AVOGARDO/self.temperature

        bond_mean = self.stats_dict['Distances']['mean'][:self.n_beads-1]
        angle_mean = self.stats_dict['Angles']['mean']

        bond_var = self.stats_dict['Distances']['std'][:self.n_beads-1]**2
        angle_var = self.stats_dict['Angles']['std']**2

        bond_keys = self.descriptions['Distances'][:self.n_beads-1]
        angle_keys = self.descriptions['Angles']

        K_bond = 1/bond_var/self.kb
        K_angle = 1/angle_var/self.kb

        bondconst_keys = np.sum([bond_keys, angle_keys])
        bondconst_array = np.vstack([np.concatenate([K_bond, K_angle]),
                                     np.concatenate([bond_mean, angle_mean])])
        if not tensor:
            bondconst_array = torch.from_numpy(bondconst_array)

        if as_dict:
            if zscores:
                bondconst_dict = self.get_zscores(tensor=tensor, as_dict=True,
                                                  order=order)
                bondconst_dict['k'] = dict(zip(bondconst_keys,
                                               bondconst_array[0, :]))
            else:
                bondconst_dict = {}
                for i, stat in enumerate(['k', 'mean']):
                    bondconst_dict[stat] = dict(zip(bondconst_keys,
                                                    bondconst_array[i, :]))
            if flip_dict:
                bondconst_dict = self._flip_dict(bondconst_dict)
            return bondconst_dict
        else:
            return bondconst_array

    def _get_distance_indices(self):
        """Determines indices of pairwise distance features
        """
        order = []
        adj_pairs = []
        for increment in range(1, self.data.shape[1]):
            for i in range(self.data.shape[1] - increment):
                order.append((i, i+increment))
                if increment == 1:
                    adj_pairs.append((i, i+increment))
        self._order = order
        self._adj_pairs = adj_pairs

    def _get_stats(self, X, key):
        """Populates stats dictionary with mean and std of feature
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        var = np.var(X, axis=0)
        self.stats_dict[key] = {}
        self.stats_dict[key]['mean'] = mean
        self.stats_dict[key]['std'] = std

    def _get_pairwise_distances(self):
        """Obtain pairwise distances for all pairs of beads;
           shape=(n_frames, n_beads-1)
        """
        dlist = np.empty([self.n_frames,
                          len(self._order)])
        for frame in range(self.n_frames):
            dmat = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(self.data[frame]))
            frame_dists = [dmat[self._order[i]]
                           for i in range(len(self._order))]
            dlist[frame, :] = frame_dists
        self.distances = dlist
        self.descriptions['Distances'] = self._order

    def _get_adjacent_distances(self):
        """Obtain adjacent distances; shape=(n_frames, n_beads-1, 3)
        """
        self.adj_dists = self.data[:][:, 1:] - \
            self.data[:][:, :(self.n_beads-1)]

    def _get_angles(self):
        """Obtain angles of all adjacent triplets; shape=(n_frames, n_beads-2)
        """
        self._get_adjacent_distances()
        base = self.adj_dists[:, 0:(self.n_beads-2), :]
        offset = self.adj_dists[:, 1:(self.n_beads-1), :]

        descriptions = []
        self.angles = np.arccos(np.sum(base*offset, axis=2)/np.linalg.norm(
            base, axis=2)/np.linalg.norm(offset, axis=2))
        descriptions.extend([(i, i+1, i+2) for i in range(self.n_beads-2)])
        self.descriptions['Angles'] = descriptions

    def _get_dihedrals(self):
        """Obtain angles of all adjacent quartets; shape=(n_frames, n_beads-3)
        """
        self._get_adjacent_distances()

        base = self.adj_dists[:, 0:(self.n_beads-2), :]
        offset = self.adj_dists[:, 1:(self.n_beads-1), :]
        offset_2 = self.adj_dists[:, 1:(self.n_beads-2), :]

        cross_product_adj = np.cross(base, offset, axis=2)
        cp_base = cross_product_adj[:, 0:(self.n_beads-3), :]
        cp_offset = cross_product_adj[:, 1:(self.n_beads-2), :]

        plane_vector = np.cross(cp_offset, offset_2, axis=2)
        pv_base = plane_vector[:, 0:(self.n_beads-3), :]

        descriptions = []
        self.dihedral_cosines = np.sum(cp_base*cp_offset, axis=2)/np.linalg.norm(
            cp_base, axis=2)/np.linalg.norm(cp_offset, axis=2)

        self.dihedral_sines = np.sum(cp_base*pv_base, axis=2)/np.linalg.norm(
            cp_base, axis=2)/np.linalg.norm(pv_base, axis=2)
        descriptions.extend([(i, i+1, i+2, i+3)
                             for i in range(self.n_beads-3)])
        self.descriptions['Dihedral_cosines'] = descriptions
        self.descriptions['Dihedral_sines'] = descriptions
