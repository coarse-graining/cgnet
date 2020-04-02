# Author: Brooke Husic
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def multi_molecule_collate(inputs):
    """ This function is used to construct padded batches for datasets
    that consist of proteins of different bead numbers. This must be
    done because tensors passed through neural networks must all
    be the same size. It must be assigned to the keyword
    argument 'collate_fn' in a PyTorch DataLoader object when working
    with variable size inputs to the network.

    Parameters
    ----------
    inputs : list of dictionaries
        This is the input list of unpadded data examples. Each example
        is a dictionary with the following key/value pairs:

            'coords' : np.array of shape (1, num_beads, 3)
            'forces' : np.array of shape (1, num_beads, 3)
            'embed'  : np.array of shape (num_beads)

    Returns
    -------
    batch : tuple of torch.tensors
        All the data in the batch, padded according to the largest system
        in the batch. The orer of tensors in the tuple is teh following:

            coords, forces, embedding_property = batch

    Notes
    -----
    See docs in MultiMoleculeDataset. While this function pads the inputs
    to the model, It is imoprtant to properly mask padded portions of tensors
    that are passed to the model. If these padded portions are not masked,
    then their artifical contribution carries through to the
    calculation of forces from the energy and the evaluation of the
    model loss. In particular, for MSE-style losses, there is a
    backpropagation instability associated with square root operations
    evaluated at 0.
    """

    embeddings = pad_sequence([torch.tensor(input['embeddings'])
                               for input in inputs], batch_first=True)
    coordinates =  pad_sequence([torch.tensor(input['coords'],
                                 requires_grad=True)
                                 for input in inputs], batch_first=True)
    forces =  pad_sequence([torch.tensor(input['forces'])
                            for input in inputs], batch_first=True)
    return coordinates, forces, embeddings


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

    def __init__(self, coordinates, forces, embeddings=None, selection=None,
                 stride=1, device=torch.device('cpu')):
        self.stride = stride

        self.coordinates = self._make_array(coordinates, selection)
        self.forces = self._make_array(forces, selection)
        if embeddings is not None:
            if (np.any(embeddings < 1) or
                not np.all(embeddings.astype(int) == embeddings)):
                raise ValueError("Embeddings must be positive integers.")
            self.embeddings = self._make_array(embeddings, selection)
        else:
            self.embeddings = None

        self._check_size_consistency()
        self.len = len(self.coordinates)
        self.device = device

    def __getitem__(self, index):
        """This will always return 3 items: coordinates, forces, embeddings.
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


class MultiMoleculeDataset(Dataset):
    """Dataset object for organizing data from molecules of differing sizes.
    It is meant to be paired with multi_molecule_collate function for use in
    a PyTorch DataLoader object. With this collating function, the inputs to
    the model will be padded on an example-by-example basis so that batches
    of tensors all have a single aggregated shape before being passed into
    the model.

    Parameters
    ----------
    coordinates: list of numpy.arrays
        List of individual examples, each corresponding to a numpy array of
        of shape [n_beads, 3], containing the cartesian coordinates of a
        single frame for that molecule
    forces: list of numpy.arrays
        List of individual examples, each corresponding to a numpy array of
        of shape [n_beads, 3], containing the cartesian forces of a
        single frame for that molecule
    embeddings: list of numpy.arrays
        List of individual examples, each corresponding to a numpy array of
        of shape [n_beads], containing the bead embeddings of a
        single frame for that molecule

    Attributes
    ----------
    data: list of dictionaries
        List of individual examples for molecules of different sizes. Each
        example is a dictionary with the following key/value pairs:

            'coords' : np.array of size [n_beads, 3]
            'forces' : np.array of size [n_beads, 3]
            'embed'  : np.array of size [n_beads]

    Example
    -------
    my_dataset = MultiMoleculeDataset(coords, forces, embeddings)
    my_loader = torch.utils.data.DataLoader(my_dataset, batch_size=512,
                                            collate_fn = multi_protein_collate,
                                            shuffle = True)

    """

    def __init__(self, coordinates, forces, embeddings=None, selection=None,
                 stride=1, device=torch.device('cpu')):
        if not (len(coordinates) == len(forces) == len(embeddings)):
            raise ValueError("Coordinates, forces, and embeddings must "
                             " contain the same number of examples")
        self.len = len(coordinates)
        self._make_data_array(coordinates, forces, embeddings)

    def __getitem__(self, indices):
        """Returns the indices of examples. Meant to be paired with
        the collating function multi_molecule_protein()
        """
        return self.data[indices]

    def __len__(self):
        return self.len

    def _make_data_array(self, coordinates, forces, embeddings, selection=None):
        """Assemble the NumPy arrays into a list of individual dictionaries"""
        self.data = []
        for idx in range(self.len):
            self.data.append({
                "coords" : coordinates[idx],
                "forces" : forces[idx],
                "embeddings" : embeddings[idx]
            })
