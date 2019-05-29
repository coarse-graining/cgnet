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

    def get_zscores(self, tensor=True,
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

        zscore_array = np.vstack([
            np.concatenate([self.stats_dict[key][stat]
                            for key in order]) for stat in ['mean', 'std']])
        if tensor:
            return torch.from_numpy(zscore_array)
        else:
            return zscore_array

    def get_bond_constants(self, tensor=True):
        """Obtain bond constants (K values and means) for adjacent distance
           and angle features. K values depend on the temperature.

        Parameters
        ----------
        tensor : Boolean (default=True)
            Returns type torch.Tensor if True and np.array if False

        Returns
        -------
        bondconst_array : torch.Tensor or np.array
            2 by n tensor/array with bond constants in the first row and
            means in the second row, where n is the number of adjacent
            pairwise distances plus the number of angles
        """
        if self.distances is None or self.angles is None:
            raise RuntimeError('Must compute distances and angles in \
                                order to get bond constants')

        self.kb = JPERKCAL/KBOLTZMANN/AVOGARDO/self.temperature

        bond_mean = self.stats_dict['Distances']['mean'][:self.n_beads-1]
        bond_var = self.stats_dict['Distances']['std'][:self.n_beads-1]**2
        angle_mean = self.stats_dict['Angles']['mean']
        angle_var = self.stats_dict['Angles']['std']**2

        K_bond = 1/bond_var/self.kb
        K_angle = 1/angle_var/self.kb

        bondconst_array = np.vstack([np.concatenate([K_bond, K_angle]),
                                     np.concatenate([bond_mean, angle_mean])])
        if tensor:
            return torch.from_numpy(bondconst_array)
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
        self.order = order
        self.adj_pairs = adj_pairs

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
                          len(self.order)])
        for frame in range(self.n_frames):
            dmat = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(self.data[frame]))
            frame_dists = [dmat[self.order[i]] for i in range(len(self.order))]
            dlist[frame, :] = frame_dists
        self.distances = dlist

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

        self.angles = np.arccos(np.sum(base*offset, axis=2)/np.linalg.norm(
            base, axis=2)/np.linalg.norm(offset, axis=2))

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

        self.dihedral_cosines = np.sum(cp_base*cp_offset, axis=2)/np.linalg.norm(
            cp_base, axis=2)/np.linalg.norm(cp_offset, axis=2)

        self.dihedral_sines = np.sum(cp_base*pv_base, axis=2)/np.linalg.norm(
            cp_base, axis=2)/np.linalg.norm(pv_base, axis=2)
