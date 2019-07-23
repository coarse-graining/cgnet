# Authors: Brooke Husic, Nick Charron
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial
import warnings


KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184


class ProteinBackboneStatistics():
    """Calculation of statistics for protein backbone features; namely
   distances, angles, and dihedral cosines and sines.

    Parameters
    ----------
    data : torch.Tensor or np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    custom_features : list of tuples or None (default=None)
    backbone_inds : 'all', list or np.ndarray, or None (default='all')
        Which bead indices correspond to consecutive beads along the backbone
    get_all_distances : Boolean (default=True)
        Whether to calculate all pairwise distances
    get_backbone_angles : Boolean (default=True)
        Whether to calculate angles along the backbone
    get_backbone_dihedrals : Boolean, (default=True)
        Whether to calculate dihedral cosines and sines along the backbone
    temperature : float (default=300.0)
        Temperature of system
    get_redundant_distance_mapping : Boolean (default=True)
        If true, creates a redundant_distance_mapping attribute

    Attributes
    ----------
    stats_dict : dictionary
        Stores 'mean' and 'std' for caluclated features
    descriptions : dictionary
        List of indices (value) for each feature type (key)
    redundant_distance_mapping
        Redundant square distance matrix

    Example
    -------
    ds = ProteinBackboneStatistics(data, n_beads = 10)
    print(ds.stats_dict['Distances']['mean'])
    """

    def __init__(self, data, custom_features=None, backbone_inds='all',
                 get_all_distances=True, get_backbone_angles=True,
                 get_backbone_dihedrals=True, temperature=300.0,
                 get_redundant_distance_mapping=True):
        if torch.is_tensor(data):
            self.data = data.detach().numpy()
        else:
            self.data = data

        self.n_frames = self.data.shape[0]
        self.n_beads = self.data.shape[1]
        self.temperature = temperature

        if custom_features is not None:
            if (np.min([len(feat) for feat in custom_features]) < 2 or
                np.max([len(feat) for feat in custmo_features]) > 4):
                raise ValueError(
                "Custom features must be tuples of length 2, 3, or 4."
                    )
            self._custom_distances = [feat for feat in custom_features if len(feat) == 2]
            self._custom_angles = [feat for feat in custom_features if len(feat) == 3]
            self._custom_dihedrals = [feat for feat in custom_features if len(feat) == 4]

        if type(backbone_inds) is str:
            if backbone_inds == 'all':
                self.backbone_inds = np.arange(self.n_beads)
                self._backbone_map = {ind : ind for ind in range(self.n_beads)}
        elif type(backbone_inds) in [list, np.ndarray]:
            if len(np.unique(backbone_inds)) != len(backbone_inds):
                raise ValueError('Backbone is not allowed to have repeat entries')
            self.backbone_inds = np.array(backbone_inds)

            if not np.all(np.sort(self.backbone_inds) == self.backbone_inds):
                warnings.warn(
    "Your backbone indices aren't sorted: Make sure your backbone indices are in consecutive order"
                               )

            self._backbone_map = self._get_backbone_map()
        elif backbone_inds is None:
            self.backbone_inds = np.array([])
        else:
            raise RuntimeError(
                "backbone_inds must be list or np.ndarray of indices, 'all', or None"
            )
        self.n_backbone_beads = len(self.backbone_inds)

        self._get_distance_indices()
        self.stats_dict = {}

        self.distances = []
        self._adj_backbone_dists = []
        self.angles = []
        self.dihedral_cosines = []
        self.dihedral_sines = []

        self._name_dict = {
            'Distances': self.distances,
            'Angles': self.angles,
            'Dihedral_cosines': self.dihedral_cosines,
            'Dihedral_sines': self.dihedral_sines
        }
        self.descriptions = {
            'Distances': [],
            'Angles': [],
            'Dihedral_cosines': [],
            'Dihedral_sines': []
        }

        self.order = []

        if get_all_distances:
            self._get_all_pairwise_distances()
            self._name_dict['Distances'] = self.distances
            #self._get_stats(self.distances, 'Distances')
            self.order += ['Distances']
            if get_redundant_distance_mapping:
                self._get_redundant_distance_mapping()

        if get_backbone_angles:
            self._get_backbone_angles()
            self._name_dict['Angles'].append(self.backbone_angles)
            #self._get_stats(self.angles, 'Angles')
            self.order += ['Angles']

        if get_backbone_dihedrals:
            self._get_backbone_dihedrals()
            self._name_dict['Dihedral_cosines'].append(self.backbone_dihedral_cosines)
            self._name_dict['Dihedral_sines'].append(self.backbone_dihedral_sines)
            #self._get_stats(self.dihedral_cosines, 'Dihedral_cosines')
            #self._get_stats(self.dihedral_sines, 'Dihedral_sines')
            self.order += ['Dihedral_cosines']
            self.order += ['Dihedral_sines']

    def _get_backbone_map(self):
        backbone_map = {mol_ind: bb_ind for bb_ind, mol_ind
                        in enumerate(self.backbone_inds)}
        pad_map = {mol_ind: np.nan for mol_ind
                   in range(self.n_beads) if mol_ind not in self.backbone_inds}
        return {**backbone_map, **pad_map}

    def _get_key(self, key, name):
        if name == 'Dihedral_cosines':
            return tuple(list(key) + ['cos'])
        if name == 'Dihedral_sines':
            return tuple(list(key) + ['sin'])
        else:
            return key

    def _flip_dict(self, mydict):
        all_inds = np.unique(np.concatenate([list(mydict[stat].keys())
                                             for stat in mydict.keys()]))

        newdict = {}
        for i in all_inds:
            newdict[i] = {}
            for stat in mydict.keys():
                if i in mydict[stat].keys():
                    newdict[i][stat] = mydict[stat][i]
        return newdict

    def get_zscores(self, tensor=True, as_dict=True, flip_dict=True):
        """Obtain zscores (mean and standard deviation) for features

        Parameters
        ----------
        tensor : Boolean (default=True)
            Returns (innermost data) of type torch.Tensor if True and np.array
             if False
        as_dict : Boolean (default=True)
            Returns a dictionary instead of an array (see "Returns"
            documentation)
        flip_dict : Boolean (default=True)
            Returns a dictionary with outer keys as indices if True and
            outer keys as statistic string names if False

        Returns
        -------
        zscore_dict : python dictionary (if as_dict=True)
            If flip_dict is True, the outer keys will be bead pairs, triples,
            or quadruples+phase, e.g. (1, 2) or (0, 1, 2, 3, 'cos'), and
            the inner keys will be 'mean' and 'std' statistics.
            If flip_dict is False, the outer keys will be the 'mean' and 'std'
            statistics and the inner keys will be bead pairs, triples, or
            quadruples+phase
        zscore_array : torch.Tensor or np.array (if as_dict=False)
            2 by n tensor/array with means in the first row and
            standard deviations in the second row, where n is
            the number of features
        """
        for key in self.order:
            if self._name_dict[key] is None:
                raise ValueError("{} have not been calculated".format(key))

        zscore_keys = np.sum([[self._get_key(key, name)
                               for key in self.descriptions[name]]
                              for name in self.order])
        zscore_array = np.vstack([
            np.concatenate([self.stats_dict[key][stat]
                            for key in self.order]) for stat in ['mean', 'std']])
        if tensor:
            zscore_array = torch.from_numpy(zscore_array).float()

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
                           flip_dict=True):
        """Obtain bond constants (K values and means) for adjacent distance
           and angle features. K values depend on the temperature.

        Parameters
        ----------
        tensor : Boolean (default=True)
            Returns (innermost data) of type torch.Tensor if True and np.array
             if False
        as_dict : Boolean (default=True)
            Returns a dictionary instead of an array (see "Returns"
            documentation)
        zscores : Boolean (default=True)
            Includes results from the get_zscores() method if True;
            only allowed if as_dict is also True
        flip_dict : Boolean (default=True)
            Returns a dictionary with outer keys as indices if True and
            outer keys as statistic string names if False

        Returns
        -------
        bondconst_dict : python dictionary (if as_dict=True)
            If flip_dict is True, the outer keys will be bead pairs, triples,
            or quadruples+phase, e.g. (1, 2) or (0, 1, 2, 3, 'cos'), and
            the inner keys will be 'mean', 'std', and 'k' statistics (only
            'mean' and 'k', and no quadruples, if zscores is False)
            If flip_dict is False, the outer keys will be the 'mean', 'std',
            and 'k' statistics (only 'mean' and 'k' if zscores if False) and
            the inner keys will be bead pairs, triples, or, unless zscores is
            False, quadruples+phase
        bondconst_array : torch.Tensor or np.array (if as_dict=False)
            2 by n tensor/array with bond constants in the first row and
            means in the second row, where n is the number of adjacent
            pairwise distances plus the number of angles
        """
        if zscores and not as_dict:
            raise RuntimeError('zscores can only be True if as_dict is True')

        if self.distances is None or self.angles is None:
            raise RuntimeError(
                'Must compute distances and angles in order to get bond constants'
            )

        self.beta = JPERKCAL/KBOLTZMANN/AVOGADRO/self.temperature

        bond_mean = self.stats_dict['Distances']['mean'][:self.n_beads-1]
        angle_mean = self.stats_dict['Angles']['mean']

        bond_var = self.stats_dict['Distances']['std'][:self.n_beads-1]**2
        angle_var = self.stats_dict['Angles']['std']**2

        bond_keys = self.descriptions['Distances'][:self.n_beads-1]
        angle_keys = self.descriptions['Angles']

        K_bond = 1/bond_var/self.beta
        K_angle = 1/angle_var/self.beta

        bondconst_keys = np.sum([bond_keys, angle_keys])
        bondconst_array = np.vstack([np.concatenate([K_bond, K_angle]),
                                     np.concatenate([bond_mean, angle_mean])])
        if not tensor:
            bondconst_array = torch.from_numpy(bondconst_array)

        if as_dict:
            if zscores:
                bondconst_dict = self.get_zscores(tensor=tensor, as_dict=True,
                                                  flip_dict=False)
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
        pair_order = []
        adj_backbone_pairs = []
        for increment in range(1, self.data.shape[1]):
            for i in range(self.data.shape[1] - increment):
                pair_order.append((i, i+increment))
                if self.backbone_inds is not None:
                    if (self._backbone_map[i+increment]
                        - self._backbone_map[i] == 1):
                        adj_backbone_pairs.append((i, i+increment))
        self._pair_order = pair_order
        self._adj_backbone_pairs = adj_backbone_pairs

    def _get_stats(self, X, key):
        """Populates stats dictionary with mean and std of feature
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        var = np.var(X, axis=0)
        self.stats_dict[key] = {}
        self.stats_dict[key]['mean'] = mean
        self.stats_dict[key]['std'] = std

    def _get_all_pairwise_distances(self):
        """Obtain pairwise distances for all pairs of beads;
           shape=(n_frames, n_beads-1)
        """
        dlist = np.empty([self.n_frames,
                          len(self._pair_order)])
        for frame in range(self.n_frames):
            dmat = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(self.data[frame]))
            frame_dists = [dmat[self._pair_order[i]]
                           for i in range(len(self._pair_order))]
            dlist[frame, :] = frame_dists
        self.distances = dlist
        self.descriptions['Distances'] = self._pair_order

    def _get_adjacent_backbone_distances(self):
        """Obtain adjacent backbone distances; 
        shape=(n_frames, n_backbone_beads-1, 3)
        """
        # self.adj_dists = self.data[:][:, 1:] - \
        #     self.data[:][:, :(self.n_beads-1)]
        self._adj_backbone_dists = (self.data[:, self.backbone_inds[1:]] -
                                   self.data[:, self.backbone_inds[:-1]])

    def _get_backbone_angles(self):
        """Obtain angles of all adjacent triplets;
        shape=(n_frames, n_backbone_beads-2)
        """
        if len(self._adj_backbone_dists) == 0:
            self._get_adjacent_backbone_distances()

        base = self._adj_backbone_dists[:, 0:(self.n_backbone_beads-2), :]
        offset = self._adj_backbone_dists[:, 1:(self.n_backbone_beads-1), :]

        descriptions = []
        self.backbone_angles = np.arccos(np.sum(base*offset, axis=2)/np.linalg.norm(
            base, axis=2)/np.linalg.norm(offset, axis=2))
        descriptions.extend([(self.backbone_inds[i], self.backbone_inds[i+1],
                              self.backbone_inds[i+2])
                              for i in range(self.n_backbone_beads-2)])
        self.descriptions['Angles'].extend(descriptions)

    def _get_backbone_dihedrals(self):
        """Obtain angles of all adjacent quartets;
        shape=(n_frames, n_backbone_beads-3)
        """
        if len(self._adj_backbone_dists) == 0:
            self._get_adjacent_backbone_distances()

        base = self._adj_backbone_dists[:, 0:(self.n_backbone_beads-2), :]
        offset = self._adj_backbone_dists[:, 1:(self.n_backbone_beads-1), :]
        offset_2 = self._adj_backbone_dists[:, 1:(self.n_backbone_beads-2), :]

        cross_product_adj = np.cross(base, offset, axis=2)
        cp_base = cross_product_adj[:, 0:(self.n_backbone_beads-3), :]
        cp_offset = cross_product_adj[:, 1:(self.n_backbone_beads-2), :]

        plane_vector = np.cross(cp_offset, offset_2, axis=2)
        pv_base = plane_vector[:, 0:(self.n_backbone_beads-3), :]

        descriptions = []
        self.backbone_dihedral_cosines = np.sum(cp_base*cp_offset, axis=2)/np.linalg.norm(
            cp_base, axis=2)/np.linalg.norm(cp_offset, axis=2)

        self.backbone_dihedral_sines = np.sum(cp_base*pv_base, axis=2)/np.linalg.norm(
            cp_base, axis=2)/np.linalg.norm(pv_base, axis=2)
        descriptions.extend([(self.backbone_inds[i], self.backbone_inds[i+1],
                              self.backbone_inds[i+2], self.backbone_inds[i+3])
                              for i in range(self.n_backbone_beads-3)])
        self.descriptions['Dihedral_cosines'].extend(descriptions)
        self.descriptions['Dihedral_sines'].extend(descriptions)

    def return_indices(self, feature_type):
        """Return all indices for specified feature type. Useful for
        constructing priors or other layers that make callbacks to
        a subset of features output from a ProteinBackboneFeature()
        layer

        Parameters
        ----------
        feature_type : str in {'Distances', 'Bonds', 'Angles',
                               'Dihedral_sines', 'Dihedral_cosines'}
            specifies for which feature type the indices should be returned

        Returns
        -------
        indices : list(int)
            list of integers corresponding the indices of specified features
            output from a ProteinBackboneFeature() layer.

        """
        if feature_type not in self.descriptions.keys() and feature_type != 'Bonds':
            raise RuntimeError(
                "Error: \'{}\' is not a valid backbone feature.".format(feature_type))
        nums = [len(self.descriptions[i]) for i in self.order]
        start_idx = 0
        for num, desc in zip(nums, self.order):
            if feature_type == desc or (feature_type == 'Bonds'
                                        and desc == 'Distances'):
                break
            else:
                start_idx += num
        if feature_type == 'Bonds': # TODO
            indices = [self.descriptions['Distances'].index(pair)
                       for pair in self._adj_backbone_pairs]
        if feature_type != 'Bonds':
            indices = range(0, len(self.descriptions[feature_type]))
        indices = [idx + start_idx for idx in indices]
        return indices

    def _get_redundant_distance_mapping(self):
        """Reformulates pairwise distances from shape [n_examples, n_dist]
        to shape [n_examples, n_beads, n_neighbors]

        This is done by finding the index mapping between non-redundant and
        redundant representations of the pairwise distances. This mapping can
        then be supplied to Schnet-related features, such as a
        RadialBasisFunction() layer, which use redundant pairwise distance
        representations.

        """
        pairwise_dist_inds = [zipped_pair[1] for zipped_pair in sorted(
                                [z for z in zip(self._pair_order,
                                                np.arange(len(self._pair_order)))
                                 ])
                            ]
        map_matrix = scipy.spatial.distance.squareform(pairwise_dist_inds)
        map_matrix = map_matrix[~np.eye(map_matrix.shape[0],
                                        dtype=bool)].reshape(
                                            map_matrix.shape[0], -1)
        self.redundant_distance_mapping = map_matrix


def kl_divergence(dist_1, dist_2):
    r"""Compute the Kullback-Leibler (KL) divergence between two discrete
    distributions according to:

    \sum_i P_i \log(P_i / Q_i)

    where P_i is the reference distribution and Q_i is the test distribution

    Parameters
    ----------
    dist_1 : numpy.array
        reference distribution of shape [n,] for n points
    dist_2 : numpy.array
        test distribution of shape [n,] for n points

    Returns
    -------
    divergence : float
        the Kullback-Leibler divergence of the two distributions

    Notes
    -----
    The KL divergence is not symmetric under distribution exchange;
    the expectation is taken over the reference distribution.

    """
    if len(dist_1) != len(dist_2):
        raise ValueError('Distributions must be of equal length')

    dist_1m = np.ma.masked_where(dist_1 == 0, dist_1)
    dist_2m = np.ma.masked_where(dist_2 == 0, dist_2)
    summand = dist_1m * np.ma.log(dist_1m / dist_2m)
    divergence = np.ma.sum(summand)
    return divergence


def js_divergence(dist_1, dist_2):
    r"""Compute the Jenson-Shannon (JS) divergence between two discrete
    distributions according to:

    0.5 * \sum_i P_i \log(P_i / M_i) + 0.5 * \sum_i Q_i \log(Q_i / M_i),

    where M_i is the elementwise mean of P_i and Q_i. This is equivalent to,

    0.5 * kl_divergence(P, Q) + 0.5 * kl_divergence(Q, P).

    Parameters
    ----------
    dist_1 : numpy.array
        first distribution of shape [n,] for n points
    dist_2 : numpy.array
        second distribution of shape [n,] for n points

    Returns
    -------
    divergence : float
        the Jenson-Shannon divergence of the two distributions

    Notes
    -----
    The JS divergence is the symmetrized extension of the KL divergence.
    It is also referred to as the information radius.

    References
    ----------
    Lin, J. (1991). Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory.
        https://dx.doi.org/10.1109/18.61115

    """
    if len(dist_1) != len(dist_2):
        raise ValueError('Distributions must be of equal length')

    dist_1m = np.ma.masked_where(dist_1 == 0, dist_1)
    dist_2m = np.ma.masked_where(dist_2 == 0, dist_2)
    elementwise_mean = 0.5 * (dist_1m + dist_2m)
    divergence = (0.5*kl_divergence(dist_1m, elementwise_mean) +
                  0.5*kl_divergence(dist_2m, elementwise_mean))
    return divergence


def histogram_intersection(dist_1, dist_2, bins=None):
    """Compute the intersection between two histograms

    Parameters
    ----------
    dist_1 : numpy.array
        first distribution of shape [n,] for n points
    dist_2 : numpy.array
        second distribution of shape [n,] for n points
    bins : None or numpy.array (default=None)
        bins for both dist1 and dist2; must be identical for both
        distributions of shape [k,] for k bins. If None,
        uniform bins are assumed

    Returns
    -------
    intersect : float
        The intersection of the two histograms; i.e., the overlapping density
    """
    if len(dist_1) != len(dist_2):
        raise ValueError('Distributions must be of equal length')
    if bins is not None and len(dist_1) + 1 != len(bins):
        raise ValueError('Bins length must be 1 more than distribution length')

    if bins is None:
        intervals = np.repeat(1/len(dist_1), len(dist_1))
    else:
        intervals = np.diff(bins)

    dist_1m = np.ma.masked_where(dist_1*dist_2 == 0, dist_1)
    dist_2m = np.ma.masked_where(dist_1*dist_2 == 0, dist_2)

    intersection = np.ma.multiply(np.ma.min([dist_1m, dist_2m], axis=0),
                                  intervals).sum()
    return intersection
