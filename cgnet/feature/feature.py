# Author: Brooke Husic
# Contributors: Jiang Wang


import torch
import torch.nn as nn
import numpy as np

from .geometry import Geometry
g = Geometry(method='torch')


class ProteinBackboneFeature(nn.Module):
    """Featurization of a protein backbone into pairwise distances,
    angles, and dihedrals.

    Attributes
    ----------
    n_beads : int
        Number of beads in the coarse-graining, assumed to be consecutive
        along a protein backbone
    descriptions : dictionary
        List of indices (value) for each feature type (key)
    description_order : list
        List of order of features in output vector of forward method
    distances : torch.Tensor
        List of pairwise distances according to descriptions['Distances']
    angles : torch.Tensor
        List of three-bead angles according to descriptions['Angles']
    dihedrals : torch.Tensor
        List of four-bead torsions according to descriptions['Torsions']
    """

    def __init__(self):
        super(ProteinBackboneFeature, self).__init__()

    def compute_distances(self):
        """Computes all pairwise distances."""
        distances = []
        descriptions = []
        for j in range(self.n_beads-1):
            new_distances = (self._coordinates[:, (j+1):self.n_beads, :]
                             - self._coordinates[:, 0:(self.n_beads-(j+1)), :])
            descriptions.extend([(0+i, j+1+i)
                                 for i in range(self.n_beads - (j+1))])
            distances.append(new_distances)
            if j == 0:
                self._adjacent_distances = new_distances
        distances = torch.cat(distances, dim=1)
        self.distances = torch.norm(distances, dim=2)
        self.descriptions["Distances"] = descriptions

    def compute_angles(self, data):
        """Computes all planar angles."""
        self.angles = g.get_angles(self._angles, data)
        self.descriptions["Angles"] = self._angles
        # descriptions = []
        # self.angles = torch.acos(torch.sum(
        #     self._adjacent_distances[:, 0:(self.n_beads-2), :] *
        #     self._adjacent_distances[:, 1:(self.n_beads-1), :], dim=2)/torch.norm(
        #     self._adjacent_distances[:, 0:(self.n_beads-2), :], dim=2)/torch.norm(
        #     self._adjacent_distances[:, 1:(self.n_beads-1), :], dim=2))
        # descriptions.extend([(i, i+1, i+2) for i in range(self.n_beads-2)])
        # self.descriptions["Angles"] = descriptions

    def compute_dihedrals(self, data):
        """Computes all four-term dihedral (torsional) angles."""
        (self.dihedral_cosines,
         self.dihedral_sines) = g.get_dihedrals(self._dihedrals, data)
        self.descriptions["Dihedral_cosines"] = self._dihedrals
        self.descriptions["Dihedral_sines"] = self._dihedrals
        # descriptions = []
        # cross_product_adjacent = torch.cross(
        #     self._adjacent_distances[:, 0:(self.n_beads-2), :],
        #     self._adjacent_distances[:, 1:(self.n_beads-1), :],
        #     dim=2)

        # plane_vector = torch.cross(
        #     cross_product_adjacent[:, 1:(self.n_beads-2)],
        #     self._adjacent_distances[:, 1:(self.n_beads-2), :], dim=2)

        # self.dihedral_cosines = torch.sum(
        #     cross_product_adjacent[:, 0:(self.n_beads-3), :] *
        #     cross_product_adjacent[:, 1:(self.n_beads-2), :], dim=2)/torch.norm(
        #     cross_product_adjacent[:, 0:(self.n_beads-3), :], dim=2)/torch.norm(
        #     cross_product_adjacent[:, 1:(self.n_beads-2), :], dim=2)

        # self.dihedral_sines = torch.sum(
        #     cross_product_adjacent[:, 0:(self.n_beads-3), :] *
        #     plane_vector[:, 0:(self.n_beads-3), :], dim=2)/torch.norm(
        #     cross_product_adjacent[:, 0:(self.n_beads-3), :], dim=2)/torch.norm(
        #     plane_vector[:, 0:(self.n_beads-3), :], dim=2)
        # descriptions.extend([(i, i+1, i+2, i+3)
        #                      for i in range(self.n_beads-3)])
        # self.descriptions["Dihedrals"] = descriptions

    def forward(self, data, feature_inds=[]):
        """Obtain differentiable feature

        Parameters
        ----------
        data : torch.Tensor
            Must be of dimensions [n_frames, n_beads, n_dimensions]

        Returns
        -------
        out : torch.Tensor
            Differentiable feature tensor
        """
        n = len(data)

        self._coordinates = data
        self.n_beads = data.shape[1]

        if len(feature_inds) == 0:
            self._distances, _ = g.get_distance_indices(self.n_beads)
            self._angles = [(i, i+1, i+2) for i in range(self.n_beads-2)]
            self._dihedrals = [(i, i+1, i+2, i+3) for i in range(self.n_beads-3)]

        else:  
            if (np.min([len(feat) for feat in feature_inds]) < 2 or
                np.max([len(feat) for feat in feature_inds]) > 4):
                raise ValueError(
                    "Custom features must be tuples of length 2, 3, or 4."
                    )

            self._distances = [feat for feat in feature_inds if len(feat) == 2]
            self._angles = [feat for feat in feature_inds if len(feat) == 3]
            self._dihedrals = [feat for feat in feature_inds if len(feat) == 4]

        self.descriptions = {}
        self.description_order = []

        self.compute_distances() # TODO
        out = self.distances
        self.description_order.append('Distances')

        if len(self._angles) > 0:
            self.compute_angles(data)
            out = torch.cat((out, self.angles), dim=1)
            self.description_order.append('Angles')

        if self.n_beads > 3:
            self.compute_dihedrals(data)
            out = torch.cat((out,
                             self.dihedral_cosines,
                             self.dihedral_sines), dim=1)
            self.description_order.append('Dihedral_cosines')
            self.description_order.append('Dihedral_sines')

        return out
