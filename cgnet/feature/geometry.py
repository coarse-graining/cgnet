# Author: Brooke Husic

import torch
import numpy as np

class Geometry():
    def __init__(self, method='torch'):
        if method == 'torch':
            self.setup_torch()
        elif method == 'numpy':
            self.setup_numpy()
        else:
            raise RuntimeError("Allowed methods are 'torch' and 'numpy'")

    def setup_torch(self):
        self.arccos = torch.acos
        self.norm = torch.norm
        self.sum = torch.sum

        self.cross = lambda x, y, axis : torch.cross(x, y, dim=axis)

    def setup_numpy(self):
        self.arccos = np.arccos
        self.norm = np.linalg.norm
        self.sum = np.sum

        self.cross = lambda x, y, axis : np.cross(x, y, axis=axis)

    def get_angle_inputs(self, angle_inds, data):
        """TODO
        """
        ind_list = [[feat[i] for feat in angle_inds]
                    for i in range(3)]

        dist_list = [data[:, ind_list[i+1], :]
                     - data[:, ind_list[i], :]
                     for i in range(2)]

        return dist_list

    def get_angles(self, angle_inds, data):
        """TODO
        """
        base, offset = self.get_angle_inputs(angle_inds, data)

        angles = self.arccos(self.sum(base*offset, axis=2)/self.norm(
                                base, axis=2)/self.norm(offset, axis=2))

        return angles

    def get_dihedrals(self, dihed_inds, data):
        """TODO
        """
        angle_inds = np.concatenate([[(f[i], f[i+1], f[i+2])
                                 for i in range(2)] for f in dihed_inds])
        base, offset = self.get_angle_inputs(angle_inds, data)
        offset_2 = base[:,1:]

        cross_product_adj = self.cross(base, offset, axis=2)
        cp_base = cross_product_adj[:, :-1, :]
        cp_offset = cross_product_adj[:, 1:, :]

        plane_vector = self.cross(cp_offset, offset_2, axis=2)

        dihedral_cosines = self.sum(cp_base[:,::2]*cp_offset[:,::2],
                                  axis=2)/self.norm(
            cp_base[:,::2], axis=2)/self.norm(cp_offset[:,::2], axis=2)

        dihedral_sines = self.sum(cp_base[:,::2]*plane_vector[:,::2],
                                axis=2)/self.norm(
            cp_base[:,::2], axis=2)/self.norm(plane_vector[:,::2], axis=2)

        return dihedral_cosines, dihedral_sines









