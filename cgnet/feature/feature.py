# Author: Brooke Husic, Dominik Lemm
# Contributors: Jiang Wang


import warnings

import numpy as np
import torch
import torch.nn as nn
from cgnet.feature.utils import ShiftedSoftplus, LinearLayer
import numpy as np
import warnings


from cgnet.feature.utils import ShiftedSoftplus, LinearLayer
from .geometry import Geometry
from .utils import RadialBasisFunction

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
    """

    def __init__(self, feature_tuples='all', n_beads=None):
        super(GeometryFeature, self).__init__()

        self._n_beads = n_beads
        if feature_tuples is not 'all':
            _temp_dict = dict(
                zip(feature_tuples, np.arange(len(feature_tuples))))
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
        out = torch.Tensor([])

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
    n_gaussians: int
        Number of Gaussians that has been used in the radial basis function.
        Needed to determine the input feature size of the first dense layer.
    n_filters: int
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

    def __init__(self, n_gaussians, n_filters):
        super(ContinuousFilterConvolution, self).__init__()
        filter_layers = LinearLayer(n_gaussians, n_filters, bias=True,
                                    activation=ShiftedSoftplus())
        # No activation function in the last layer allows the filter generator
        # to contain negative values.
        filter_layers += LinearLayer(n_filters, n_filters, bias=True)
        self.filter_generator = nn.Sequential(*filter_layers)

    def forward(self, features, rbf_expansion, neighbor_list):
        """ Compute convolutional block

        Parameters
        ----------
        features: torch.Tensor
            Feature vector of size [n_frames, n_beads, n_features].
        rbf_expansion: torch.Tensor
            Gaussian expansion of bead distances of size
            [n_frames, n_beads, n_neighbors, n_gaussians].
        neighbor_list: torch.Tensor
            Indices of all neighbors of each bead.
            Size [n_frames, n_beads, n_neighbors]

        Returns
        -------
        aggregated_features: torch.Tensor
            Residual features of size [n_frames, n_beads, n_features]

        """

        # Generate the convolutional filter
        # Shape (n_frames, n_beads, n_neighbors, n_features)
        conv_filter = self.filter_generator(rbf_expansion)

        # Feature tensor needs to be transformed from
        # (n_frames, n_beads, n_features)
        # to
        # (n_frames, n_beads, n_neighbors, n_features)
        # This can be done by feeding the features of a respective bead into
        # its position in the neighbor_list.
        n_batch, n_beads, n_neighbors = neighbor_list.size()

        # Shape (n_frames, n_beads * n_neighbors, 1)
        neighbor_list = neighbor_list.reshape(-1, n_beads * n_neighbors, 1)
        # Shape (n_frames, n_beads * n_neighbors, n_features)
        neighbor_list = neighbor_list.expand(-1, -1, features.size(2))

        # Gather the features into the respective places in the neighbor list
        neighbor_features = torch.gather(features, 1, neighbor_list)
        # Reshape back to (n_frames, n_beads, n_neighbors, n_features) for
        # element-wise multiplication with the filter
        neighbor_features = neighbor_features.reshape(n_batch, n_beads,
                                                      n_neighbors, -1)

        # Element-wise multiplication of the features with
        # the convolutional filter
        conv_features = neighbor_features * conv_filter

        # Aggregate/pool the features from (n_frames, n_beads, n_neighs, n_feats)
        # to (n_frames, n_beads, n_features)
        aggregated_features = torch.sum(conv_features, dim=2)
        return aggregated_features


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
    n_inputs: int
        Number of input features. Determines input size for the initial linear
        layer.
    n_gaussians: int
        Number of Gaussians that has been used in the radial basis function.
        Needed in to determine the input size of the continuous filter
        convolution.
    n_filters: int
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

    def __init__(self, n_inputs, n_gaussians, n_filters):
        super(InteractionBlock, self).__init__()

        self.inital_dense = nn.Sequential(
            *LinearLayer(n_inputs, n_filters, bias=False,
                         activation=None))
        self.cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                                  n_filters=n_filters)
        output_layers = LinearLayer(n_filters, n_filters, bias=True,
                                    activation=ShiftedSoftplus())
        output_layers += LinearLayer(n_filters, n_filters, bias=True,
                                     activation=None)
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, features, rbf_expansion, neighbor_list):
        """ Compute interaction block

        Parameters
        ----------
        features: torch.Tensor
            Input features from an embedding or interaction layer.
            Shape [n_frames, n_beads, n_features]
        rbf_expansion: torch.Tensor
            Radial basis function expansion of inter-bead distances.
            Shape [n_frames, n_beads, n_neighbors, n_gaussians]
        neighbor_list: torch.Tensor
            Indices of all neighbors of each bead.
            Size [n_frames, n_beads, n_neighbors]

        Returns
        -------
        output_features: torch.Tensor
            Output of an interaction block. This output can be used to form
            a residual connection with the output of a prior embedding/interaction
            layer.
            Shape [n_frames, n_beads, n_filters]

        """
        init_feature_output = self.inital_dense(features)
        conv_output = self.cfconv(init_feature_output, rbf_expansion,
                                  neighbor_list)
        output_features = self.output_dense(conv_output)
        return output_features


class SchnetFeature(nn.Module):
    """Wrapper class for radial basis function layer, continuous filter convolution,
    and interaction block connecting feature inputs and outputs residuallly
    """

    def __init__(self,
                 feature_size,
                 embedding_layer,
                 calculate_geometry=None,
                 n_beads=None,
                 rbf_cutoff=5.0,
                 n_gaussians=50,
                 variance=1.0,
                 n_interaction_blocks=1,
                 share_weights=False):
        """

        Parameters
        ----------
        feature_size: int
            Defines the number of neurons of the linear layers in the
            InteractionBlock. Also defines the number of convolutional
            filters that will be used.
        embedding_layer: torch.nn.Module
            Class that embeds a property into a feature vector.
        calculate_geometry: boolean (default=False)
            Allows calls to Geometry instance for calculating distances for a
            standalone SchnetFeature instance (i.e. one that is not
            preceded by a GeometryFeature instance).
        n_beads: int (default=None)
            Number of coarse grain beads in the model.
        rbf_cutoff: float (default=5.0)
            Cutoff for the radial basis function.
        n_gaussians: int (default=50)
            Number of gaussians for the gaussian expansion in the radial basis
            function.
        variance: float (default=1.0)
            The variance (standard deviation squared) of the Gaussian functions.
        n_interaction_blocks: int (default=1)
            Number of interaction blocks.
        share_weights: bool (default=False)
            If True, shares the weights between all interaction blocks.

        Notes
        -----
        Default values for radial basis function related variables (rbf_cutoff,
        n_gaussians, variance) are taken as suggested in SchnetPack.

        Example
        -------
        # TODO: Maybe add an exmaple here once it's functional?

        """
        super(SchnetFeature, self).__init__()
        self.embedding_layer = embedding_layer
        self.rbf_layer = RadialBasisFunction(cutoff=rbf_cutoff,
                                             n_gaussians=n_gaussians,
                                             variance=variance)
        if share_weights:
            # Lets the interaction blocks share the weights
            self.interaction_blocks = nn.ModuleList(
                [InteractionBlock(feature_size, n_gaussians, feature_size)]
                * n_interaction_blocks
            )
        else:
            # Every interaction block has their own weights
            self.interaction_blocks = nn.ModuleList(
                [InteractionBlock(feature_size, n_gaussians, feature_size)
                 for _ in range(n_interaction_blocks)]
            )

        self.n_beads = n_beads
        self.calculate_geometry = calculate_geometry
        if self.calculate_geometry:
            self._distance_pairs, _ = g.get_distance_indices(n_beads, [], [])
            self.redundant_distance_mapping = g.get_redundant_distance_mapping(
                self._distance_pairs)
        else:
            self._distance_pairs = None
            self.redundant_distance_mapping = None

    def forward(self, in_features, embedding_property):
        """Forward method through single Schnet block

        Parameters
        ----------
        in_features: torch.Tensor (grad enabled)
            input trajectory/data of size [n_frames, n_in_features].
        embedding_property: torch.Tensor
            Some property that should be embedded. Can be nuclear charge
            or maybe an arbitrary number assigned for amino-acids.
            Size [n_frames, n_properties]

        Returns
        -------
        features: torch.Tensor
            Atom-wise feature representation.
            Size [n_frames, n_beads, n_features]

        """
        # if geometry is specified, the distances are calculated from input
        # coordinates. Otherwise, it is assumed that in_features are
        # pairwise distances in redundant form
        if self.calculate_geometry:
            distances = g.get_distances(self._distance_pairs, in_features,
                                        norm=True)
            distances = distances[:, self.redundant_distance_mapping]
        # TODO compute neighborlist
        neighbors = self.compute_neighbors(coordinates)

        features = self.embedding_layer(embedding_property)
        rbf_expansion = self.rbf_layer(distances=distances)

        for interaction_block in self.interaction_blocks:
            interaction_features = interaction_block(features=features,
                                                     rbf_expansion=rbf_expansion,
                                                     neighbor_list=neighbors)
            features = features + interaction_features

        return features


class CGBeadEmbedding(torch.nn.Module):
    def __init__(self, n_embeddings, embedding_dim):
        """Simple embedding class for coarse-grain beads.
        Serves as a lookup table that returns a fixed size embedding.

        Parameters
        ----------
        n_embeddings: int
            Maximum number of different properties/amino_acids/elements,
            basically the dictionary size.
        embedding_dim: int
            Size of the embedding vector.
        """
        super(CGBeadEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

    def forward(self, embedding_property):
        """

        Parameters
        ----------
        embedding_property: torch.Tensor
            Some property that should be embedded. Can be nuclear charge
            or maybe an arbitrary number assigned for amino-acids. Passing a
            zero will produce an embedding vector filled with zeroes (necessary
            in the case of zero padded batches).
            Size [n_frames, n_properties]

        Returns
        -------
        embedding_vector: torch.Tensor
            Corresponding embedding vector to the passed indices.
        """
        return self.embedding(embedding_property)
