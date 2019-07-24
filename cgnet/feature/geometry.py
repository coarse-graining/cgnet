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

    def setup_numpy(self):
        self.arccos = np.arccos
        self.norm = np.linalg.norm
        self.sum = np.sum

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
        base, offset = self.get_angle_inputs(angle_inds, data)

        angles = self.arccos(self.sum(base*offset, axis=2)/self.norm(
                                base, axis=2)/self.norm(offset, axis=2))

        return angles

