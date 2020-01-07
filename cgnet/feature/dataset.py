# Author: Brooke Husic
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
    embeddings : np.array
        Embedding data of dimension [n_frames, n_beads, n_embedding_properties]
        Embeddings must be positive integers.
    selection : np.array (default=None)
        Array of frame indices to select from the coordinates and forces.
        If None, all are used.
    stride : int (default=1)
        Subsample the data by 1 / stride.
    device : torch.device (default=torch.device('cpu'))
        CUDA device/GPU on which to mount tensors drawn from __getitem__().
        Default device is the local CPU.
    """

    def __init__(self, coordinates, forces, embeddings=None, dummy_atoms=False,
                 selection=None, stride=1, device=torch.device('cpu')):
        self.stride = stride

        self.coordinates = self._make_array(coordinates, selection)
        self.forces = self._make_array(forces, selection)
        if embeddings is not None:
            if not np.all(embeddings.astype(int) == embeddings):
                raise ValueError("Embeddings must be integers.")
            if np.any(embeddings < 1) and not dummy_atoms:
                raise ValueError("Embeddings must be positive.")
            self.embeddings = self._make_array(embeddings, selection)
        else:
            self.embeddings = None

        self._check_size_consistency()
        self.len = len(self.coordinates)
        self.device = device

    def __getitem__(self, index):
        """This will always return 3 items: coordinates, frames, embeddings.
        If embeddings are not given, then the third object returned will
        be an empty tensor.
        """
        if self.embeddings is None:
            # Still returns three objects, but the third is an empty tensor
            return (
                torch.tensor(self.coordinates[index],
                             requires_grad=True, device=self.device),
                torch.tensor(self.forces[index],
                             device=self.device),
                torch.tensor([])
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
        """We add data to the dataset with a custom selection and the stride
        specified upon object instantiation, ensuring that the embeddings
        have a shape length of 2, and that everything has the same number
        of frames.
        """
        new_coords = self._make_array(coordinates, selection)
        new_forces = self._make_array(forces, selection)
        if embeddings is not None:
            new_embeddings = self._make_array(embeddings, selection)

        self.coordinates = np.concatenate(
            [self.coordinates, new_coords], axis=0)
        self.forces = np.concatenate([self.forces, new_forces], axis=0)

        if self.embeddings is not None:
            self.embeddings = np.concatenate([self.embeddings, new_embeddings],
                                             axis=0)
        self._check_size_consistency()
        self.len = len(self.coordinates)

    def _check_size_consistency(self):
        """When we create or add data, we need to make sure that everything
        has the same number of frames.
        """
        if self.coordinates.shape != self.forces.shape:
            raise ValueError("Coordinates and forces must have equal shapes")

        if len(self.coordinates.shape) != 3:
            raise ValueError("Coordinates and forces must have three dimensions")

        if self.embeddings is not None:
            if len(self.embeddings.shape) != 2:
                raise ValueError("Embeddings must have two dimensions")
