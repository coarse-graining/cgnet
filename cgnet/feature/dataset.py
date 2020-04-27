# Author: Brooke Husic, Nick Charron
# Contributors: Jiang Wang


import numpy as np
import torch
import scipy.spatial

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def multi_molecule_collate(input_dictionaries, device=torch.device('cpu')):
    """This function is used to construct padded batches for datasets
    that consist of molecules of different bead numbers. This must be
    done because tensors passed through neural networks must all
    be the same size. This method must be passed to the 'collate_fn'
    keyword argument in a PyTorch DataLoader object when working
    with variable size inputs to a network (see example below).

    Parameters
    ----------
    input_dictionaries : list of dictionaries
        This is the input list of *unpadded* input data. Each example in the
        list is a dictionary with the following key/value pairs:

            'coords' : np.array of shape (1, num_beads, 3)
            'forces' : np.array of shape (1, num_beads, 3)
            'embed'  : np.array of shape (num_beads)

        Embeddings must be specified for this function to work correctly.
        A KeyError will be raised if they are not.

    Returns
    -------
    batch : tuple of torch.tensors
        All the data in the batch, padded according to the largest system
        in the batch. The orer of tensors in the tuple is the following:

            coords, forces, embedding_property = batch

        All examples are right-padded with zeros. For example, if the
        maximum bead size in list of examples is 8, the embedding array
        for an example from a molecule composed of 3 beads will be padded
        from:

            upadded_embedding = [1, 2, 5]

        to:

            padded_embedding = [1, 2, 5, 0, 0, 0, 0, 0]

        An analogous right-aligned padding is done for forces and
        coordinates.

    Notes
    -----
    See docs in MultiMoleculeDataset. While this function pads the inputs
    to the model, It is important to properly mask padded portions of tensors
    that are passed to the model. If these padded portions are not masked,
    then their artifical contribution carries through to the
    calculation of forces from the energy and the evaluation of the
    model loss. In particular, for MSE-style losses, there is a
    backpropagation instability associated with square root operations
    evaluated at 0.

    Example
    -------
    my_loader = torch.utils.data.DataLoader(my_dataset, batch_size=512,
                                            collate_fn=multi_molecule_collate,
                                            shuffle=True)
    """

    coordinates =  pad_sequence([torch.tensor(example['coords'],
                                 requires_grad=True, device=device)
                                 for example in input_dictionaries],
                                 batch_first=True)
    forces = pad_sequence([torch.tensor(example['forces'], device=device)
                           for example in input_dictionaries],
                           batch_first=True)
    embeddings = pad_sequence([torch.tensor(example['embeddings'], device=device)
                               for example in input_dictionaries],
                               batch_first=True)
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

        self._check_inputs()

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

        self._check_inputs()

        self.len = len(self.coordinates)

    def _check_inputs(self):
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

            if self.coordinates.shape[0] != self.embeddings.shape[0]:
                raise ValueError("Embeddings must have the same number of examples "
                                 "as coordinates/forces")

            if self.coordinates.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Embeddings must have the same number of beads "
                                 "as the coordinates/forces")


class MultiMoleculeDataset(Dataset):
    """Dataset object for organizing data from molecules of differing sizes.
    It is meant to be paired with multi_molecule_collate function for use in
    a PyTorch DataLoader object. With this collating function, the inputs to
    the model will be padded on an example-by-example basis so that batches
    of tensors all have a single aggregated shape before being passed into
    the model.

    Note that unlike MoleculeDataset, MultiMoleculeDataset takes a *list*
    of numpy arrays.

    Parameters
    ----------
    coordinates_list: list of numpy.arrays
        List of coordinate data. Each item i in the list must be a numpy
        array of shape [n_beads_i, 3], containing the cartesian coordinates of
        a single frame for molecule i
    forces_list: list of numpy.arrays
        List of force data. Each item i in the list must be a numpy
        array of shape [n_beads_i, 3], containing the cartesian forces of a
        single frame for molecule i
    embeddings_list: list of numpy.arrays
        List of embeddings. Each item i in the list must be a numpy array
        of shape [n_beads_i], containing the bead embeddings of a
        single frame for molecule i. The embedding_list may not be None;
        MultiMoleculeDataset is only compatible with SchnetFeatures.

    Attributes
    ----------
    data: list of dictionaries
        List of individual examples for molecules of different sizes. Each
        example is a dictionary with the following key/value pairs:

            'coords' : np.array of size [n_beads_i, 3]
            'forces' : np.array of size [n_beads_i, 3]
            'embed'  : np.array of size [n_beads_i]

    Example
    -------
    my_multi_dataset = MultiMoleculeDataset(list_of_coords, list_of_forces,
                                            list_of_embeddings)
    my_loader = torch.utils.data.DataLoader(my_multi_dataset, batch_size=512,
                                            collate_fn=multi_molecule_collate,
                                            shuffle=True)

    """

    def __init__(self, coordinates_list, forces_list, embeddings_list,
                 selection=None, stride=1, device=torch.device('cpu')):
        self._check_inputs(coordinates_list, forces_list,
                           embeddings_list=embeddings_list)
        self.stride = stride
        self.data = None

        self._make_array_data(coordinates_list, forces_list,
                              embeddings_list=embeddings_list,
                              selection=selection)
        self.len = len(self.data)

    def __getitem__(self, indices):
        """Returns the list of examples corresponding to the supplied indices. It
        is meant to be paired with the collating function multi_molecule_collate()
        """
        if isinstance(indices, int):
            return self.data[indices]
        else:
            return [self.data[i] for i in indices]

    def __len__(self):
        return self.len

    def _make_array_data(self, coordinates_list, forces_list,
                         embeddings_list, selection=None):
        """Assemble the NumPy arrays into a list of individual dictionaries for
        use with the multi_molecule_collate function.
        """

        if self.data == None:
            self.data = []
        if selection is not None:
            coordinates = [coordinates_list[i] for i in selection]
            forces = [forces_list[i] for i in selection]
            embeddings = [embeddings_list[i] for i in selection]
            for coord, force, embed in zip(coordinates[::self.stride],
                                           forces[::self.stride],
                                           embeddings[::self.stride]):
                self.data.append({
                    "coords" : coord, "forces" : force, "embeddings" : embed})
        else:
            for coord, force, embed in zip(coordinates_list[::self.stride],
                                           forces_list[::self.stride],
                                           embeddings_list[::self.stride]):
                self.data.append({
                    "coords" : coord, "forces" : force, "embeddings" : embed})


    def add_data(self, coordinates_list, forces_list, embeddings_list,
                 selection=None):
        """We add data to the dataset with a custom selection and the stride
        specified upon object instantiation, ensuring that the embeddings
        have a shape length of 1, and that everything has the same number
        of frames.
        """
        self._check_inputs(coordinates_list, forces_list,
                           embeddings_list=embeddings_list)
        self._make_array_data(coordinates_list, forces_list,
                              embeddings_list=embeddings_list, selection=selection)
        self.len = len(self.data)

    def _check_inputs(self, coordinates_list, forces_list, embeddings_list):
        """Helper function for ensuring data has the correct shape when
        adding examples to a MultiMoleculeDataset. This function also checks to
        to make sure that no embeddings are 0.
        """

        if embeddings_list is None:
            raise ValueError("Embeddings must be supplied, as MultiMoleculeDataset"
                             " is intended to be used only with SchNet utilities.")
        else:
            for embedding in embeddings_list:
                if np.any(embedding < 1):
                    raise ValueError("Embeddings must be positive integers.")

        if not (len(coordinates_list) == len(forces_list) == len(embeddings_list)):
            raise ValueError("Coordinates, forces, and embeddings lists must "
                             " contain the same number of examples")

        for idx, (coord, force, embed) in enumerate(zip(coordinates_list, forces_list,
                                                        embeddings_list)):
            if coord.shape != force.shape:
                raise ValueError("Coordinates and forces must have equal shapes at example", idx)

            if len(coord.shape) != 2:
                raise ValueError("Coordinates and forces must have two dimensions at example", idx)

            if len(embed.shape) != 1:
                raise ValueError("Embeddings must have one dimension at example", idx)

            if coord.shape[0] != embed.shape[0]:
                raise ValueError("Embeddings must have the same number of beads "
                                 "as the coordinates/forces at example", idx)
