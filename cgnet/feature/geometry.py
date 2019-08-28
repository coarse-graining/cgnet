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
    """

    def __init__(self, method='torch'):
        self.method = method
        if method == 'torch':
            self.setup_torch()
        elif method == 'numpy':
            self.setup_numpy()
        else:
            raise RuntimeError("Allowed methods are 'torch' and 'numpy'")

    def setup_torch(self):
        self.arccos = torch.acos

        self.cross = lambda x, y, axis: torch.cross(x, y, dim=axis)
        self.norm = lambda x, axis: torch.norm(x, dim=axis)
        self.sum = lambda x, axis: torch.sum(x, dim=axis)

    def setup_numpy(self):
        self.arccos = np.arccos

        self.cross = lambda x, y, axis: np.cross(x, y, axis=axis)
        self.norm = lambda x, axis: np.linalg.norm(x, axis=axis)
        self.sum = lambda x, axis: np.sum(x, axis=axis)

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
        distances = self.get_vectorize_inputs(distance_inds, data)
        if norm:
            distances = self.norm(distances, axis=2)
        return distances

    def get_angles(self, angle_inds, data):
        """Calculates angles in a vectorized fashion.
        """
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
        n_frames, n_beads, n_neighbors = distances.shape

        # Create a simple neighbor list of shape [n_frames, n_beads, n_neighbors]
        # in which every bead sees each other but themselves.
        # First, create a matrix that contains all indices.
        neighbors = np.tile(np.arange(n_beads), (n_frames, n_beads, 1))
        # To remove the self interaction of beads, an inverted identity matrix
        # is used to exclude the respective indices in the neighbor list.
        neighbors = neighbors[:, ~np.eye(n_beads, dtype=np.bool)].reshape(
            n_frames,
            n_beads,
            n_neighbors)

        if cutoff is not None:
            if isinstance(distances, torch.Tensor):
                distances = distances.numpy()
            # Create an index mask for neighbors that are inside the cutoff
            neighbor_mask = (distances < cutoff).astype(np.float32)
            # Set the indices of beads outside the cutoff to 0
            neighbors[~neighbor_mask.astype(np.bool)] = 0
        else:
            neighbor_mask = np.ones((n_frames, n_beads, n_neighbors),
                                    dtype=np.float32)

        if self.method == 'torch':
            return torch.from_numpy(neighbors), torch.from_numpy(neighbor_mask)
        else:
            return neighbors, neighbor_mask
