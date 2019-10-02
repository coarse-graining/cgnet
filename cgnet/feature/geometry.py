# Author: Brooke Husic
# Contributors: Dominik Lemm

import numpy as np
import scipy
import torch


class Geometry():
    """Helper class to calculate distances, angles, and dihedrals
    with a unified, vectorized framework depending on whether pytorch
    or numpy is used.

    Parameters
    ----------
    method : 'torch' or 'numpy' (default='torch')
        Library used for compuations
    device : torch.device (default=torch.device('cpu'))
        Device upon which geometrical calculations will take place. When
        embedded as an attribute for a feature class, the device will inherit
        from the feature device attribute
    """

    def __init__(self, method='torch', device=torch.device('cpu')):
        self.device = device
        if method not in ['torch', 'numpy']:
            raise RuntimeError("Allowed methods are 'torch' and 'numpy'")
        self.method = method

        # # # # # # # # # # # # #
        # Define any types here #
        # # # # # # # # # # # # #
        if method == 'torch':
            self.bool = torch.bool
            self.float32 = torch.float32

        elif self.method == 'numpy':
            self.bool = np.bool
            self.float32 = np.float32

    def check_array_vs_tensor(self, object, name=None):
        if name is None:
            name = ''

        if self.method == 'numpy' and type(object) is not np.ndarray:
            raise ValueError(
    "Input argument {} must be type np.ndarray for Geometry(method='numpy')".format(name)
                )
        if self.method == 'torch' and type(object) is not torch.Tensor:
            raise ValueError(
    "Input argument {} must be type torch.Tensor for Geometry(method='torch')".format(name)
                )

    def get_distance_indices(self, n_beads, backbone_inds=[], backbone_map=None):
        """Determines indices of pairwise distance features.
        """
        pair_order = []
        adj_backbone_pairs = []
        for increment in range(1, n_beads):
            for i in range(n_beads - increment):
                pair_order.append((i, i+increment))
                if len(backbone_inds) > 0:
                    if (backbone_map[i+increment]
                            - backbone_map[i] == 1):
                        adj_backbone_pairs.append((i, i+increment))

        return pair_order, adj_backbone_pairs

    def get_redundant_distance_mapping(self, pair_order):
        """Reformulates pairwise distances from shape [n_frames, n_dist]
        to shape [n_frames, n_beads, n_neighbors]

        This is done by finding the index mapping between non-redundant and
        redundant representations of the pairwise distances. This mapping can
        then be supplied to Schnet-related features, such as a
        RadialBasisFunction() layer, which use redundant pairwise distance
        representations.

        """
        pairwise_dist_inds = [zipped_pair[1] for zipped_pair in sorted(
            [z for z in zip(pair_order,
                            np.arange(len(pair_order)))
             ])
        ]
        map_matrix = scipy.spatial.distance.squareform(pairwise_dist_inds)
        map_matrix = map_matrix[~np.eye(map_matrix.shape[0],
                                        dtype=bool)].reshape(
                                            map_matrix.shape[0], -1)
        return map_matrix

    def get_vectorize_inputs(self, inds, data):
        """Helper function to obtain indices for vectorized calculations.
        """
        if len(np.unique([len(feat) for feat in inds])) > 1:
            raise ValueError(
                "All features must be the same length."
            )
        feat_length = len(inds[0])

        ind_list = [[feat[i] for feat in inds]
                    for i in range(feat_length)]

        dist_list = [data[:, ind_list[i+1], :]
                     - data[:, ind_list[i], :]
                     for i in range(feat_length - 1)]

        if len(dist_list) == 1:
            dist_list = dist_list[0]

        return dist_list

    def get_distances(self, distance_inds, data, norm=True):
        """Calculates distances in a vectorized fashion.
        """
        self.check_array_vs_tensor(data, 'data')

        distances = self.get_vectorize_inputs(distance_inds, data)
        if norm:
            distances = self.norm(distances, axis=2)
        return distances

    def get_angles(self, angle_inds, data):
        """Calculates angles in a vectorized fashion.
        """
        self.check_array_vs_tensor(data, 'data')

        base, offset = self.get_vectorize_inputs(angle_inds, data)

        angles = self.arccos(self.sum(base*offset, axis=2)/self.norm(
            base, axis=2)/self.norm(offset, axis=2))

        return angles

    def get_dihedrals(self, dihed_inds, data):
        """Calculates dihedrals in a vectorized fashion.

        Note
        ----
        This is implemented in a hacky/bad way. It calculates twice as many
        dihedrals as needed and removes every other one. There is a better
        way to do this, I think using two lists of angles, but for now
        this has the correct functionality.
        """
        self.check_array_vs_tensor(data, 'data')

        angle_inds = np.concatenate([[(f[i], f[i+1], f[i+2])
                                      for i in range(2)] for f in dihed_inds])
        base, offset = self.get_vectorize_inputs(angle_inds, data)
        offset_2 = base[:, 1:]

        cross_product_adj = self.cross(base, offset, axis=2)
        cp_base = cross_product_adj[:, :-1, :]
        cp_offset = cross_product_adj[:, 1:, :]

        plane_vector = self.cross(cp_offset, offset_2, axis=2)

        dihedral_cosines = self.sum(cp_base[:, ::2]*cp_offset[:, ::2],
                                    axis=2)/self.norm(
            cp_base[:, ::2], axis=2)/self.norm(cp_offset[:, ::2], axis=2)

        dihedral_sines = self.sum(cp_base[:, ::2]*plane_vector[:, ::2],
                                  axis=2)/self.norm(
            cp_base[:, ::2], axis=2)/self.norm(plane_vector[:, ::2], axis=2)

        return dihedral_cosines, dihedral_sines

    def get_neighbors(self, distances, cutoff=None):
        """Calculates a simple neighbor list in which every bead sees
        each other. If a cutoff is specified, only beads inside that distance
        cutoff are considered as neighbors.

        Parameters
        ----------
        distances: torch.Tensor or np.array
            Redundant distance matrix of shape (n_frames, n_beads, n_neighbors).
        cutoff: float (default=None)
            Distance cutoff in Angstrom in which beads are considered neighbors.

        Returns
        -------
        neighbors: torch.Tensor or np.array
            Indices of all neighbors of each bead.
            Shape [n_frames, n_beads, n_neighbors]
        neighbor_mask: torch.Tensor or np.array
            Index mask to filter out non-existing neighbors that were
            introduced to due distance cutoffs.
            Shape [n_frames, n_beads, n_neighbors]

        """

        self.check_array_vs_tensor(distances, 'distances')

        n_frames, n_beads, n_neighbors = distances.shape

        # Create a simple neighbor list of shape [n_frames, n_beads, n_neighbors]
        # in which every bead sees each other but themselves.
        # First, create a matrix that contains all indices.
        neighbors = self.tile(self.arange(n_beads), (n_frames, n_beads, 1))
        # To remove the self interaction of beads, an inverted identity matrix
        # is used to exclude the respective indices in the neighbor list.
        neighbors = neighbors[:, ~self.eye(n_beads, dtype=self.bool)].reshape(
            n_frames,
            n_beads,
            n_neighbors)

        if cutoff is not None:
            # Create an index mask for neighbors that are inside the cutoff
            neighbor_mask = distances < cutoff
            # Set the indices of beads outside the cutoff to 0
            neighbors[~neighbor_mask] = 0
            neighbor_mask = self.to_type(neighbor_mask, self.float32)
        else:
            neighbor_mask = self.ones((n_frames, n_beads, n_neighbors),
                                      dtype=self.float32)

        return neighbors, neighbor_mask

    def _torch_eye(self, n, dtype):
        if dtype == torch.bool:
            # Only in pytorch>=1.2!
            return torch.BoolTensor(np.eye(n, dtype=np.bool))
        else:
            return torch.eye(n, dtype=dtype)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # Versatile Methods # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Methods defined: arccos, cross, norm, sum, arange, tile, eye, ones,
    #                  to_type

    def arccos(self, x):
        if self.method == 'torch':
            return torch.acos(x)
        elif self.method == 'numpy':
            return np.arccos(x)

    def cross(self, x, y, axis):
        if self.method == 'torch':
            return torch.cross(x, y, dim=axis)
        elif self.method == 'numpy':
            return np.cross(x, y, axis=axis)

    def norm(self, x, axis):
        if self.method == 'torch':
            return torch.norm(x, dim=axis)
        elif self.method == 'numpy':
            return np.linalg.norm(x, axis=axis)

    def sum(self, x, axis):
        if self.method == 'torch':
            return torch.sum(x, dim=axis)
        elif self.method == 'numpy':
            return np.sum(x, axis=axis)

    def arange(self, n):
        if self.method == 'torch':
            return torch.arange(n)
        elif self.method == 'numpy':
            return np.arange(n)

    def tile(self, x, shape):
        if self.method == 'torch':
            return x.repeat(*shape)
        elif self.method == 'numpy':
            return np.tile(x, shape)

    def eye(self, n, dtype):
    # As of pytorch 1.2.0, BoolTensors are implemented. However,
    # torch.eye does not take dtype=torch.bool on CPU devices yet.
    # Watch pytorch PR #24148 for the implementation, which would
    # enable self.eye = lambda n, dtype: torch.eye(n, dtype=dtype)
    # For now, we do this:
        if self.method == 'torch':
            return self._torch_eye(n, dtype) #.to(self.device)
        elif self.method == 'numpy':
            return np.eye(n, dtype=dtype)

    def ones(self, shape, dtype):
        if self.method == 'torch':
            return torch.ones(*shape, dtype=dtype).to(self.device)
        elif self.method =='numpy':
            return np.ones(shape, dtype=dtype)

    def to_type(self, x, dtype):
        if self.method == 'torch':
            return x.type(dtype)
        elif self.method == 'numpy':
            return x.astype(dtype)
