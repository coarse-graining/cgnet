# Author: Brooke Husic
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial
import warnings

from torch.utils.data import Dataset, DataLoader


class MoleculeDataset(Dataset):
    """Creates dataset for coordinates and forces.

    Parameters
    ----------
    coordinates : np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    forces : np.array
        Coordinate data of dimension [n_frames, n_beads, n_dimensions]
    embeddings : np.array
        Embedding data of dimension [n_frames, n_beads, n_embedding_properties]
    selection : np.array (default=None)
        Array of frame indices to select from the coordinates and forces.
        If None, all are used.
    stride : int (default=1)
        Subsample the data by 1 / stride.
    device : torch.device (default=torch.device('cpu'))
        CUDA device/GPU on which to mount tensors drawn from __getitem__().
        Default device is the local CPU.
    """

    def __init__(self, coordinates, forces, embeddings=None, selection=None,
                 stride=1, device=torch.device('cpu')):
        self.stride = stride

        self.coordinates = self._make_array(coordinates, selection)
        self.forces = self._make_array(forces, selection)
        if embeddings is not None:
            embeddings = self._check_embedding_dimension(embeddings)
            self.embeddings = self._make_array(embeddings, selection)
        else:
            self.embeddings = None

        self._check_size_consistency()
        self.len = len(self.coordinates)
        self.device = device

    def __getitem__(self, index):
        if self.embeddings is None:
            # Still returns three objects, but the third is an empty tensor
            return (
                torch.tensor(self.coordinates[index],
                             requires_grad=True, device=self.device),
                torch.tensor(self.forces[index],
                             device=self.device),
                torch.Tensor([])
            )
        else:
            return (
                torch.tensor(self.coordinates[index],
                             requires_grad=True, device=self.device),
                torch.tensor(self.forces[index],
                             device=self.device),
                torch.tensor(self.embeddings[index],
                             device=self.device)
            )

    def __len__(self):
        return self.len

    def _make_array(self, data, selection=None):
        """Returns an array that contains a selection of data
        if specified, at the stride provided.
        """
        if selection is not None:
            return np.array(data[selection][::self.stride])
        else:
            return data[::self.stride]

    def add_data(self, coordinates, forces, embeddings=None, selection=None):
        new_coords = self._make_array(coordinates, selection)
        new_forces = self._make_array(forces, selection)
        if embeddings is not None:
            new_embeddings = self._make_array(embeddings, selection)

        self.coordinates = np.concatenate(
            [self.coordinates, new_coords], axis=0)
        self.forces = np.concatenate([self.forces, new_forces], axis=0)
        if self.embeddings is not None:
            embeddings = self._check_embedding_dimension(embeddings)
            self.embeddings = np.concatenate([self.embeddings, new_embeddings],
                                             axis=0)
        self._check_size_consistency()
        self.len = len(self.coordinates)

    def _check_size_consistency(self):
        if len(self.coordinates) != len(self.forces):
            raise ValueError("Coordinates and forces must have equal lengths")

        if self.embeddings is not None:
            if len(self.coordinates) != len(self.embeddings):
                raise ValueError(
                    "Coordinates, forces, and embeddings must have equal lengths"
                )

    def _check_embedding_dimension(self, embeddings):
        if len(embeddings.shape) == 3:
            return embeddings
        else:
            old_shape = embeddings.shape
            embeddings = embeddings.reshape(*old_shape, 1)
            warnings.warn(
                "The embeddings have been reshaped from {} to {}".format(
                                                                old_shape,
                                                                embeddings.shape
                                                                )
                )
            return embeddings
