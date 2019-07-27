# Authors: Brooke Husic, Nick Charron
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial
import warnings

from .geometry import Geometry
g = Geometry(method='numpy')

KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184


class GeometryStatistics():
    """Calculation of statistics for geometric features; namely
   distances, angles, and dihedral cosines and sines.

    Parameters
    ----------
    data : torch.Tensor or np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    custom_feature_tuples : list of tuples (default=[])
        List of 2-, 3-, and 4-element tuples containing arbitrary distance,
        angle, and dihedral features to be calculated.
    backbone_inds : 'all', list or np.ndarray, or None (default='all')
        Which bead indices correspond to consecutive beads along the backbone
    get_all_distances : Boolean (default=True)
        Whether to calculate all pairwise distances
    get_backbone_angles : Boolean (default=True)
        Whether to calculate angles among adjacent beads along the backbone
    get_backbone_dihedrals : Boolean (default=True)
        Whether to calculate dihedral cosines and sines among adjacent beads
        along the backbone
    temperature : float or None (default=300.0)
        Temperature of system. Use None for dimensionless calculations.
    get_redundant_distance_mapping : Boolean (default=True)
        If true, creates a redundant_distance_mapping attribute
    bond_pairs : list of tuples (default=[])
        List of 2-element tuples containing bonded pairs
    adjacent_backbone_bonds : Boolean, (default=True)
        Whether adjacent beads along the backbone should be considered
        as bonds

    Attributes
    ----------
    beta : float
        1/(Boltzmann constant)/(temperature) if temperature is not None in
        units of kcal per mole; otherwise 1.0
    descriptions : dictionary
        List of indices (value) for each feature type (key)
    redundant_distance_mapping
        Redundant square distance matrix

    Example
    -------
    stats = GeometryStatistics(data, n_beads = 10)
    prior_stats_dict = ds.get_prior_statistics()
    """

    def __init__(self, data, custom_feature_tuples=[], backbone_inds='all',
                 get_all_distances=True, get_backbone_angles=True,
                 get_backbone_dihedrals=True, temperature=300.0,
                 get_redundant_distance_mapping=True, bond_pairs=[],
                 adjacent_backbone_bonds=True):
        if torch.is_tensor(data):
            self.data = data.detach().numpy()
        else:
            self.data = data

        self.n_frames = self.data.shape[0]
        self.n_beads = self.data.shape[1]
        assert self.data.shape[2] == 3  # dimensions
        self.temperature = temperature
        if self.temperature is not None:
            self.beta = JPERKCAL/KBOLTZMANN/AVOGADRO/self.temperature
        else:
            self.beta = 1.0

        self._process_backbone(backbone_inds)
        self._process_custom_feature_tuples(custom_feature_tuples)
        self.get_redundant_distance_mapping = get_redundant_distance_mapping

        if not get_all_distances:
            if np.any([bond_ind not in custom_feature_tuples for bond_ind in bond_pairs]):
                raise ValueError(
                    "All bond_pairs must be also in custom_feature_tuples if get_all_distances is False."
                )
        if np.any([len(bond_ind) != 2 for bond_ind in bond_pairs]):
            raise RuntimeError(
                "All bonds must be of length 2."
            )
        self._bond_pairs = bond_pairs
        self.adjacent_backbone_bonds = adjacent_backbone_bonds

        self.order = []

        self.distances = []
        self.angles = []
        self.dihedral_cosines = []
        self.dihedral_sines = []

        self.descriptions = {
            'Distances': [],
            'Angles': [],
            'Dihedral_cosines': [],
            'Dihedral_sines': []
        }

        self._stats_dict = {}

        # # # # # # #
        # Distances #
        # # # # # # #
        if get_all_distances:
            (self._pair_order,
             self._adj_backbone_pairs) = g.get_distance_indices(self.n_beads,
                                                                self.backbone_inds,
                                                                self._backbone_map)
            if len(self._custom_distance_pairs) > 0:
                warnings.warn(
                    "All distances are already being calculated, so custom distances are meaningless."
                )
                self._custom_distance_pairs = []
            self._distance_pairs = self._pair_order

            if self.adjacent_backbone_bonds:
                if np.any([bond_ind in self._adj_backbone_pairs
                           for bond_ind in self._bond_pairs]):
                    warnings.warn(
                        "Some bond indices were already on the backbone."
                    )
                    self._bond_pairs = [bond_ind for bond_ind
                                       in self._bond_pairs
                                       if bond_ind not in self._adj_backbone_pairs]
            self.bond_pairs = self._adj_backbone_pairs

        else:
            self._distance_pairs = []
            self.bond_pairs = []
        self._distance_pairs.extend(self._custom_distance_pairs)
        self.bond_pairs.extend(self._bond_pairs)

        if len(self._distance_pairs) > 0:
            self._get_distances()

        # # # # # #
        # Angles  #
        # # # # # #
        if get_backbone_angles:
            self._angle_trips = [(self.backbone_inds[i], self.backbone_inds[i+1],
                           self.backbone_inds[i+2])
                          for i in range(len(self.backbone_inds) - 2)]
            if np.any([cust_angle in self._angle_trips
                       for cust_angle in self._custom_angle_trips]):
                warnings.warn(
                    "Some custom angles were on the backbone and will not be re-calculated."
                )
                self._custom_angle_trips = [cust_angle for cust_angle
                                           in self._custom_angle_trips
                                           if cust_angle not in self._angle_trips]
        else:
            self._angle_trips = []
        self._angle_trips.extend(self._custom_angle_trips)
        if len(self._angle_trips) > 0:
            self._get_angles()

        # # # # # # #
        # Dihedrals #
        # # # # # # #
        if get_backbone_dihedrals:
            self._dihedral_quads = [(self.backbone_inds[i], self.backbone_inds[i+1],
                              self.backbone_inds[i+2], self.backbone_inds[i+3])
                             for i in range(len(self.backbone_inds) - 3)]
            if np.any([cust_dih in self._dihedral_quads
                       for cust_dih in self._custom_dihedral_quads]):
                warnings.warn(
                    "Some custom dihedrals were on the backbone and will not be re-calculated."
                )
                self._custom_dihedral_quads = [cust_dih for _custom_dihedral_quads
                                              in self._custom_dihedral_quads
                                              if cust_dih not in self._dihedral_quads]
        else:
            self._dihedral_quads = []
        self._dihedral_quads.extend(self._custom_dihedral_quads)
        if len(self._dihedral_quads) > 0:
            self._get_dihedrals()

        self.feature_tuples = []
        for feature_type in self.order:
            if feature_type != 'Dihedral_sines':
                self.feature_tuples.extend(self.descriptions[feature_type])

    def _process_custom_feature_tuples(self, custom_feature_tuples):
        """Helper function to sort custom features into distances, angles,
        and dihedrals.
        """
        if len(custom_feature_tuples) > 0:
            if (np.min([len(feat) for feat in custom_feature_tuples]) < 2 or
                    np.max([len(feat) for feat in custom_feature_tuples]) > 4):
                raise ValueError(
                    "Custom features must be tuples of length 2, 3, or 4."
                )
            if np.max([np.max(bead) for bead in custom_feature_tuples]) > self.n_beads - 1:
                raise ValueError(
                    "Bead index in at least one feature is out of range."
                )

            _temp_dict = dict(zip(custom_feature_tuples, np.arange(len(custom_feature_tuples))))
            if len(_temp_dict) < len(custom_feature_tuples):
                custom_feature_tuples = list(_temp_dict.keys())
                warnings.warn(
                    "Some custom feature tuples are repeated and have been removed."
                    )

            self._custom_distance_pairs = [
                feat for feat in custom_feature_tuples if len(feat) == 2]
            self._custom_angle_trips = [
                feat for feat in custom_feature_tuples if len(feat) == 3]
            self._custom_dihedral_quads = [
                feat for feat in custom_feature_tuples if len(feat) == 4]
        else:
            self._custom_distance_pairs = []
            self._custom_angle_trips = []
            self._custom_dihedral_quads = []

    def _get_backbone_map(self):
        """Helper function that maps bead indices to indices along the backbone
        only.

        Returns
        -------
        backbone_map : dict
            Dictionary with bead indices as keys and, as values, backbone
            indices for beads along the backbone or np.nan otherwise.
        """
        backbone_map = {mol_ind: bb_ind for bb_ind, mol_ind
                        in enumerate(self.backbone_inds)}
        pad_map = {mol_ind: np.nan for mol_ind
                   in range(self.n_beads) if mol_ind not in self.backbone_inds}
        return {**backbone_map, **pad_map}

    def _process_backbone(self, backbone_inds):
        """Helper function to obtain attributes needed for backbone atoms.
        """
        if type(backbone_inds) is str:
            if backbone_inds == 'all':
                self.backbone_inds = np.arange(self.n_beads)
                self._backbone_map = {ind: ind for ind in range(self.n_beads)}
        elif type(backbone_inds) in [list, np.ndarray]:
            if len(np.unique(backbone_inds)) != len(backbone_inds):
                raise ValueError(
                    'Backbone is not allowed to have repeat entries')
            self.backbone_inds = np.array(backbone_inds)

            if not np.all(np.sort(self.backbone_inds) == self.backbone_inds):
                warnings.warn(
                    "Your backbone indices aren't sorted. Make sure your backbone indices are in consecutive order."
                )

            self._backbone_map = self._get_backbone_map()
        elif backbone_inds is None:
            if len(custom_feature_tuples) == 0:
                raise RuntimeError(
                    "Must have either backbone or custom features.")
            self.backbone_inds = np.array([])
            self._backbone_map = None
        else:
            raise RuntimeError(
                "backbone_inds must be list or np.ndarray of indices, 'all', or None"
            )
        self.n_backbone_beads = len(self.backbone_inds)

    def _get_distances(self):
        """Obtains all pairwise distances for the two-bead indices provided.
        """
        self.distances = g.get_distances(self._distance_pairs, self.data, norm=True)
        self.descriptions['Distances'].extend(self._distance_pairs)
        self._get_stats(self.distances, 'Distances')
        self.order += ['Distances']
        if self.get_redundant_distance_mapping:
            self._get_redundant_distance_mapping()

    def _get_angles(self):
        """Obtains all planar angles for the three-bead indices provided.
        """
        self.angles = g.get_angles(self._angle_trips, self.data)

        self.descriptions['Angles'].extend(self._angle_trips)
        self._get_stats(self.angles, 'Angles')
        self.order += ['Angles']

    def _get_dihedrals(self):
        """Obtains all dihedral angles for the four-bead indices provided.
        """
        (self.dihedral_cosines,
            self.dihedral_sines) = g.get_dihedrals(self._dihedral_quads, self.data)

        self.descriptions['Dihedral_cosines'].extend(self._dihedral_quads)
        self.descriptions['Dihedral_sines'].extend(self._dihedral_quads)

        self._get_stats(self.dihedral_cosines, 'Dihedral_cosines')
        self._get_stats(self.dihedral_sines, 'Dihedral_sines')
        self.order += ['Dihedral_cosines']
        self.order += ['Dihedral_sines']

    def _get_stats(self, X, key):
        """Populates stats dictionary with mean and std of feature.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        var = np.var(X, axis=0)
        k = 1/var/self.beta
        self._stats_dict[key] = {}
        self._stats_dict[key]['mean'] = mean
        self._stats_dict[key]['std'] = std
        self._stats_dict[key]['k'] = k

    def _get_key(self, key, name):
        """Returns keys for zscore and bond constant dictionaries based on
        description names.
        """
        if name == 'Dihedral_cosines':
            return tuple(list(key) + ['cos'])
        if name == 'Dihedral_sines':
            return tuple(list(key) + ['sin'])
        else:
            return key

    def _flip_dict(self, mydict):
        """Flips the dictionary; see documentation for get_zscores or
        get_bond_constants.
        """
        all_inds = list(mydict['mean'].keys())

        newdict = {}
        for i in all_inds:
            newdict[i] = {}
            for stat in mydict.keys():
                if i in mydict[stat].keys():
                    newdict[i][stat] = mydict[stat][i]
        return newdict

    def get_prior_statistics(self, tensor=True, as_dict=True, flip_dict=True):
        """Obtain prior statistics (mean, standard deviation, and
        bond/angle/dihedral constants) for features

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
        prior_statistics_dict : python dictionary (if as_dict=True)
            If flip_dict is True, the outer keys will be bead pairs, triples,
            or quadruples+phase, e.g. (1, 2) or (0, 1, 2, 3, 'cos'), and
            the inner keys will be 'mean' and 'std' statistics.
            If flip_dict is False, the outer keys will be the 'mean' and 'std'
            statistics and the inner keys will be bead pairs, triples, or
            quadruples+phase
        prior_statistics_array : torch.Tensor or np.array (if as_dict=False)
            2 by n tensor/array with means in the first row and
            standard deviations in the second row, where n is
            the number of features
        """
        temp_keys = [[self._get_key(key, name)
                      for key in self.descriptions[name]]
                     for name in self.order]
        prior_stat_keys = []
        for sublist in temp_keys:
            prior_stat_keys.extend(sublist)
        prior_stat_array = np.vstack([
            np.concatenate([self._stats_dict[key][stat]
                            for key in self.order])
                            for stat in ['mean', 'std', 'k']])
        if tensor:
            prior_stat_array = torch.from_numpy(prior_stat_array).float()
        self.prior_statistics_keys = prior_stat_keys
        self.prior_statistics_array = prior_stat_array

        if as_dict:
            prior_statistics_dict = {}
            for i, stat in enumerate(['mean', 'std']):
                prior_statistics_dict[stat] = dict(zip(prior_stat_keys,
                                                       prior_stat_array[i, :]))
            if flip_dict:
                prior_statistics_dict = self._flip_dict(prior_statistics_dict)
            return prior_statistics_dict
        else:
            return prior_stat_array

    def return_indices(self, feature_type):
        """Return all indices for specified feature type. Useful for
        constructing priors or other layers that make callbacks to
        a subset of features output from a GeometryFeature()
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
            output from a GeometryFeature() layer.

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
        if feature_type == 'Bonds':  # TODO
            indices = [self.descriptions['Distances'].index(pair)
                       for pair in self.bond_pairs]
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
