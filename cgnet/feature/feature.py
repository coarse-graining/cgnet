# Author: Brooke Husic, Dominik Lemm
# Contributors: Jiang Wang


import torch
import torch.nn as nn

from cgnet.feature.utils import ShiftedSoftplus
from cgnet.network.layers import LinearLayer


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

    def compute_angles(self):
        """Computes all planar angles."""
        descriptions = []
        self.angles = torch.acos(torch.sum(
            self._adjacent_distances[:, 0:(self.n_beads-2), :] *
            self._adjacent_distances[:, 1:(self.n_beads-1), :], dim=2)/torch.norm(
            self._adjacent_distances[:, 0:(self.n_beads-2), :], dim=2)/torch.norm(
            self._adjacent_distances[:, 1:(self.n_beads-1), :], dim=2))
        descriptions.extend([(i, i+1, i+2) for i in range(self.n_beads-2)])
        self.descriptions["Angles"] = descriptions

    def compute_dihedrals(self):
        """Computes all four-term dihedral (torsional) angles."""
        descriptions = []
        cross_product_adjacent = torch.cross(
            self._adjacent_distances[:, 0:(self.n_beads-2), :],
            self._adjacent_distances[:, 1:(self.n_beads-1), :],
            dim=2)

        plane_vector = torch.cross(
            cross_product_adjacent[:, 1:(self.n_beads-2)],
            self._adjacent_distances[:, 1:(self.n_beads-2), :], dim=2)

        self.dihedral_cosines = torch.sum(
            cross_product_adjacent[:, 0:(self.n_beads-3), :] *
            cross_product_adjacent[:, 1:(self.n_beads-2), :], dim=2)/torch.norm(
            cross_product_adjacent[:, 0:(self.n_beads-3), :], dim=2)/torch.norm(
            cross_product_adjacent[:, 1:(self.n_beads-2), :], dim=2)

        self.dihedral_sines = torch.sum(
            cross_product_adjacent[:, 0:(self.n_beads-3), :] *
            plane_vector[:, 0:(self.n_beads-3), :], dim=2)/torch.norm(
            cross_product_adjacent[:, 0:(self.n_beads-3), :], dim=2)/torch.norm(
            plane_vector[:, 0:(self.n_beads-3), :], dim=2)
        descriptions.extend([(i, i+1, i+2, i+3)
                             for i in range(self.n_beads-3)])
        self.descriptions["Dihedrals"] = descriptions

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

        self.descriptions = {}
        self.description_order = []

        self.compute_distances()
        out = self.distances
        self.description_order.append('Distances')

        if self.n_beads > 2:
            self.compute_angles()
            out = torch.cat((out, self.angles), dim=1)
            self.description_order.append('Angles')

        if self.n_beads > 3:
            self.compute_dihedrals()
            out = torch.cat((out,
                             self.dihedral_cosines,
                             self.dihedral_sines), dim=1)
            self.description_order.append('Dihedral_cosines')
            self.description_order.append('Dihedral_sines')

        return out


class ContinuousFilterConvolution(nn.Module):
    r"""
    Continuous-filter convolution block as described by Schütt et al. (2018).

    Unlike convential convolutional layers that utilize discrete filter tensors,
    a continuous-filter convolutional layer evaluates the convolution at discrete
    locations in space using continuous radial filters (Schütt et al. 2018).

        x_i^{l+i} = (X^i * W^l)_i = \sum_{j=0}^{n_{atoms}} x_j^l \circ W^l (r_j -r_i)
        
    with feature representation X^l=(x^l_1, ..., x^l_n), filter-generating 
    network W^l, positions R=(r_1, ..., r_n) and the current layer l.

    A continuous-filter convolution block consists of a filter generating network
    as follows:

    Filter Generator:
        1. Featurization of cartesian positions into distances
           (which are roto-translationally invariant)
           (already precomputed so will be parsed as arguments)
        2. Atom-wise/Linear layer with shifted-softplus activation function
        3. Atom-wise/Linear layer with shifted-softplus activation function
           (see Notes)

    The filter generator output is then multiplied element-wise with the
    continuous convolution filter as part of the interaction block.

    Parameters
    ----------
    num_gaussians: int
        Number of Gaussians that has been used in the radial basis function.
        Needed to determine the input feature size of the first dense layer.
    num_filters: int
        Number of filters that will be created. Also determines the output size.
        Needs to be the same size as the features of the residual connection in
        the interaction block.

    Notes
    -----
    Following the current implementation in SchNetPack, the last linear layer of
    the filter generator does not contain an activation function.
    This allows the filter generator to contain negative values.

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
    """

    def __init__(self, num_gaussians, num_filters):
        super(ContinuousFilterConvolution, self).__init__()
        filter_layers = LinearLayer(num_gaussians, num_filters, bias=True,
                                    activation=ShiftedSoftplus())
        # No activation function in the last layer allows the filter generator
        # to contain negative values.
        filter_layers += LinearLayer(num_filters, num_filters, bias=True)
        self.filter_generator = nn.Sequential(*filter_layers)

    def forward(self, features, rbf_expansion, neighbor_list):
        """ Compute convolutional block

        Parameters
        ----------
        features: torch.Tensor
            Feature vector of size [n_examples, n_beads, n_features].
        rbf_expansion: torch.Tensor
            Gaussian expansion of bead distances of size
            [n_examples, n_beads, n_neighbors, n_gaussians].
        neighbor_list: torch.Tensor
            Indices of all neighbors of each bead.
            Size [n_examples, n_beads, n_neighbors]

        Returns
        -------
        conv_features: torch.Tensor
            Residual features of size [n_examples, n_beads, n_features]

        """

        # Generate the convolutional filter
        # Shape (n_examples, n_beads, n_neighbors, n_features)
        conv_filter = self.filter_generator(rbf_expansion)

        # Feature tensor needs to be transformed from
        # (n_examples, n_beads, n_features)
        # to
        # (n_examples, n_beads, n_neighbors, n_features)
        # This can be done by feeding the features of a respective bead into
        # its position in the neighbor_list.

        num_batch, num_beads, num_neighbors = neighbor_list.size()
        # Shape (n_examples, n_beads * n_neighbors, 1)
        neighbor_list = neighbor_list.reshape(-1, num_beads * num_neighbors, 1)
        # Shape (n_examples, n_beads * n_neighbors, n_features)
        neighbor_list = neighbor_list.expand(-1, -1, features.size(2))

        # Gather the features into the respective places in the neighbor list
        neighbor_features = torch.gather(features, 1, neighbor_list)
        # Reshape back to (n_examples, n_beads, n_neighbors, n_features) for
        # element-wise multiplication with the filter
        neighbor_features = neighbor_features.reshape(num_batch, num_beads,
                                                      num_neighbors, -1)

        # Element-wise multiplication of the features with
        # the convolutional filter
        conv_features = neighbor_features * conv_filter

        # Aggregate/pool the features from (n_examples, n_beads, n_neighs, n_feats)
        # to (n_examples, n_beads, n_features)
        agg_features = torch.sum(conv_features, dim=2)
        return agg_features


class InteractionBlock(nn.Module):
    """
    SchNet interaction block as described by Schütt et al. (2018).

    An interaction block consists of:
        1. Atom-wise/Linear layer without activation function
        2. Continuous filter convolution, which is a filter-generator multiplied
           element-wise with the output of the previous layer
        3. Atom-wise/Linear layer with activation
        4. Atom-wise/Linear layer without activation

    The output of an interaction block will then be used to form an additive
    residual connection with the original input features, (x'_1, ... , x'_n),
    see Notes.

    Parameters
    ----------
    num_inputs: int
        Number of input features. Determines input size for the initial linear
        layer.
    num_gaussians: int
        Number of Gaussians that has been used in the radial basis function.
        Needed in to determine the input size of the continuous filter
        convolution.
    num_filters: int
        Number of filters that will be created in the continuous filter convolution.
        The same feature size will be used for the output linear layers of the
        interaction block.

    Notes
    -----
    The additive residual connection between interaction blocks is not
    included in the output of this forward pass. The residual connection
    will be computed separately outside of this class.

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
    """

    def __init__(self, num_inputs, num_gaussians, num_filters):
        super(InteractionBlock, self).__init__()

        self.inital_dense = nn.Sequential(
            *LinearLayer(num_inputs, num_filters, bias=False,
                         activation=None))
        self.cfconv = ContinuousFilterConvolution(num_gaussians=num_gaussians,
                                                  num_filters=num_filters)
        output_layers = LinearLayer(num_filters, num_filters, bias=True,
                                    activation=ShiftedSoftplus())
        output_layers += LinearLayer(num_filters, num_filters, bias=True,
                                     activation=None)
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, features, rbf_expansion, neighbor_list):
        """ Compute interaction block

        Parameters
        ----------
        features: torch.Tensor
            Input features from an embedding or interaction layer.
            Shape [n_examples, n_beads, n_features]
        rbf_expansion: torch.Tensor
            Radial basis function expansion of inter-bead distances.
            Shape [n_examples, n_beads, n_neighbors, n_gaussians]
        neighbor_list: torch.Tensor
            Indices of all neighbors of each bead.
            Size [n_examples, n_beads, n_neighbors]

        Returns
        -------
        output_features: torch.Tensor
            Output of an interaction block. This output can be used to form
            a residual connection with the output of a prior embedding/interaction
            layer.
            Shape [n_examples, n_beads, n_filters]

        """
        init_feature_output = self.inital_dense(features)
        conv_output = self.cfconv(init_feature_output, rbf_expansion,
                                  neighbor_list)
        output_features = self.output_dense(conv_output)
        return output_features
