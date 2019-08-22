# Author: Brooke Husic
# Contributors: Jiang Wang


import torch
import torch.nn as nn
import numpy as np
import warnings

from .geometry import Geometry
g = Geometry(method='torch')


class GeometryFeature(nn.Module):
    """Featurization of coarse-grained beads into pairwise distances,
    angles, and dihedrals.

    Parameters
    ----------
    feature_tuples : list of tuples (default=[])
        List of 2-, 3-, and 4-element tuples containing distance, angle, and
        dihedral features to be calculated.

    Attributes
    ----------
    n_beads : int
        Number of beads in the coarse-graining
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
    device : torch.device (default=torch.device('cpu'))
        Device upon which tensors are mounted. Default device is the local
        CPU.
    """

    def __init__(self, feature_tuples='all', n_beads=None, device=torch.device('cpu')):
        super(GeometryFeature, self).__init__()
        self.device = device
        self._n_beads = n_beads
        if feature_tuples is not 'all':
            _temp_dict = dict(zip(feature_tuples, np.arange(len(feature_tuples))))
            if len(_temp_dict) < len(feature_tuples):
                feature_tuples = list(_temp_dict.keys())
                warnings.warn(
                    "Some feature tuples are repeated and have been removed."
                    )

            self.feature_tuples = feature_tuples
            if (np.min([len(feat) for feat in feature_tuples]) < 2 or
                    np.max([len(feat) for feat in feature_tuples]) > 4):
                raise ValueError(
                    "Custom features must be tuples of length 2, 3, or 4."
                )

            self._distance_pairs = [
                feat for feat in feature_tuples if len(feat) == 2]
            self._angle_trips = [
                feat for feat in feature_tuples if len(feat) == 3]
            self._dihedral_quads = [
                feat for feat in feature_tuples if len(feat) == 4]
        else:
            if n_beads is None:
                raise RuntimeError(
                    "Must specify n_beads if feature_tuples is 'all'."
                    )
            self._distance_pairs, _ = g.get_distance_indices(n_beads)
            if n_beads > 2:
                self._angle_trips = [(i, i+1, i+2)
                                    for i in range(n_beads-2)]
            else:
                self._angle_trips = []
            if n_beads > 3:
                self._dihedral_quads = [(i, i+1, i+2, i+3)
                                       for i in range(n_beads-3)]
            else:
                self._dihedral_quads = []
            self.feature_tuples = self._distance_pairs + \
                self._angle_trips + self._dihedral_quads

    def compute_distances(self, data):
        """Computes all pairwise distances."""
        self.distances = g.get_distances(self._distance_pairs, data, norm=True)
        self.descriptions["Distances"] = self._distance_pairs

    def compute_angles(self, data):
        """Computes planar angles."""
        self.angles = g.get_angles(self._angle_trips, data)
        self.descriptions["Angles"] = self._angle_trips

    def compute_dihedrals(self, data):
        """Computes four-term dihedral (torsional) angles."""
        (self.dihedral_cosines,
         self.dihedral_sines) = g.get_dihedrals(self._dihedral_quads, data)
        self.descriptions["Dihedral_cosines"] = self._dihedral_quads
        self.descriptions["Dihedral_sines"] = self._dihedral_quads

    def forward(self, data):
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
        if self._n_beads is not None and self.n_beads != self._n_beads:
            raise ValueError(
                "n_beads passed to __init__ does not match n_beads in data."
                )
        if np.max([np.max(bead) for bead in self.feature_tuples]) > self.n_beads - 1:
            raise ValueError(
                "Bead index in at least one feature is out of range."
            )

        self.descriptions = {}
        self.description_order = []
        out = torch.Tensor([], device=self.device)

        if len(self._distance_pairs) > 0:
            self.compute_distances(data)
            out = torch.cat((out, self.distances), dim=1)
            self.description_order.append('Distances')
        else:
            self.distances = torch.Tensor([])

        if len(self._angle_trips) > 0:
            self.compute_angles(data)
            out = torch.cat((out, self.angles), dim=1)
            self.description_order.append('Angles')
        else:
            self.angles = torch.Tensor([])

        if len(self._dihedral_quads) > 0:
            self.compute_dihedrals(data)
            out = torch.cat((out,
                             self.dihedral_cosines,
                             self.dihedral_sines), dim=1)
            self.description_order.append('Dihedral_cosines')
            self.description_order.append('Dihedral_sines')
        else:
            self.dihedral_cosines = torch.Tensor([])
            self.dihedral_sines = torch.Tensor([])

        return out
