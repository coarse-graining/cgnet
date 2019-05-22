# Author: B Husic
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial

from torch.utils.data import Dataset, DataLoader


class MoleculeDataset(Dataset):
    """Creates dataset for coordinates and forces.

    Parameters
    ----------
    coordinates : np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    forces : np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    selection : np.array (default=None)
        Array of indices to select from the coordinates and forces.
        If None, all are used.
    stride : int (default=1)
        Subsample the data by 1 / stride.
    """

    def __init__(self, coordinates, forces, selection=None, stride=1):
        self.stride = stride
        self.coordinates, self.forces = self._make_arrays(coordinates,
                                                           forces, selection)

        self.len = len(self.coordinates)

    def __getitem__(self, index):
        return (torch.from_numpy(self.coordinates[index]),
                torch.from_numpy(self.forces[index]))

    def __len__(self):
        return self.len

    def _make_arrays(self, coordinates, forces, selection=None):
        if selection is not None:
            coordinates = np.array(coordinates[selection][::self.stride])
            forces = np.array(forces[selection][::self.stride])
        else:
            coordinates = np.array(coordinates[::self.stride])
            forces = np.array(forces[::self.stride])

        if len(coordinates) != len(forces):
            raise ValueError("Coordinates and forces must have equal lengths")

        return coordinates, forces

    def add_data(self, coordinates, forces, selection=None):
        new_coords, new_forces = self._make_arrays(coordinates, forces,
                                                    selection)
        self.coordinates = np.concatenate([self.coordinates, new_coords], axis=0)
        self.forces = np.concatenate([self.forces, new_forces], axis=0)

        self.len = len(self.coordinates)
