# Author: Brooke Husic, Dominik Lemm
# Contributors: Jiang Wang

import torch
import torch.nn as nn
import numpy as np
import warnings

from .geometry import Geometry
from .utils import (GaussianRBF, PolynomialCutoffRBF, ShiftedSoftplus)
from .schnet_utils import InteractionBlock


class GeometryFeature(nn.Module):
    """Featurization of coarse-grained beads into pairwise distances,
    angles, and dihedrals.

    Parameters
    ----------
    feature_tuples : list of tuples (default=None) or 'all_backbone'
        List of 2-, 3-, and 4-element tuples containing distance, angle, and
        dihedral features to be calculated. If 'all_backbone', then
        all beads are assumed to be the backbone, and all pairwise distances
        and consecutive three-bead angles and four-bead dihedrals are
        calculated. If 'all_backbone', then n_beads must be provided.
    n_beads : int or None (default=None)
        Number of beads; only needs to be provided if 'all_backbone' is used.
        If a list of feature_tuples are provided and n_beads is provided
        incorrectly, an error will be raised.
    device : torch.device (default=torch.device('cpu'))
        Device upon which tensors are mounted. Default device is the local
        CPU.

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
    geometry : Geometry instance
        Helper class which performs geometrical calculations. The device of
        used is that same as the feature class device.
    """

    def __init__(self, feature_tuples=None, n_beads=None,
                 device=torch.device('cpu')):
        super(GeometryFeature, self).__init__()

        self._n_beads = n_beads
        self.device = device
        self.geometry = Geometry(method='torch', device=self.device)
        if feature_tuples is not 'all_backbone':
            if feature_tuples is not None:
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
                raise RuntimeError("Either a list of feature tuples or "
                                   "'all_backbone' must be specified.")
        else:
            if n_beads is None:
                raise RuntimeError(
                    "Must specify n_beads if feature_tuples is 'all_backone'."
                )
            self._distance_pairs, _ = self.geometry.get_distance_indices(
                n_beads)
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
        self.distances = self.geometry.get_distances(self._distance_pairs,
                                                     data, norm=True)
        self.descriptions["Distances"] = self._distance_pairs

    def compute_angles(self, data):
        """Computes planar angles."""
        self.angles = self.geometry.get_angles(self._angle_trips, data)
        self.descriptions["Angles"] = self._angle_trips

    def compute_dihedrals(self, data):
        """Computes four-term dihedral (torsional) angles."""
        (self.dihedral_cosines,
         self.dihedral_sines) = self.geometry.get_dihedrals(self._dihedral_quads,
                                                            data)
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
        out = torch.Tensor([]).to(self.device)

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


class SchnetFeature(nn.Module):
    """Wrapper class for radial basis function layer, continuous filter convolution,
    and interaction block connecting feature inputs and outputs residuallly.

    Parameters
    ----------
    feature_size: int
        Defines the number of neurons of the linear layers in the
        InteractionBlock. Also defines the number of convolutional
        filters that will be used.
    embedding_layer: torch.nn.Module
        Class that embeds a property into a feature vector.
    rbf_layer : nn.Module
        Radial basis function expansions layer. This layer is used
        to expand pairwise distances in a basis prior to passing
        through the continuous filter convolution.
    n_beads: int
        Number of coarse grain beads in the model.
    activation: nn.Module (default=ShiftedSoftplus())
        Activation function that will be used throughout the
        SchnetFeature - including within the atom-wise layers and
        the filter-generating networks. Following Schütt et al,
        the default value is ShiftedSoftplus, but any differentiable
        activation function can be used (see Notes).
    calculate_geometry: boolean (default=False)
        Allows calls to Geometry instance for calculating distances for a
        standalone SchnetFeature instance (i.e. one that is not
        preceded by a GeometryFeature instance).
    neighbor_cutoff: float (default=None)
        Cutoff distance in whether beads are considered neighbors or not.
    normalization_layer: nn.Module (default=None)
        Normalization layer to be applied to the ouptut of the
        ContinuousFilterConvolution
    n_interaction_blocks: int (default=1)
        Number of interaction blocks.
    share_weights: bool (default=False)
        If True, shares the weights between all interaction blocks.
    share_batchnorm_parameters: bool (default=False)
        If True, all BatchNorm1d instances in the model share parameters.
        Note: this option is only applicable if a nn.BatchNorm1d object
        has been passed to the 'normalization_layer' kwarg. By default,
        SchnetFeature instances do not include normalization layers.

    Notes
    -----
    Default values for radial basis function related variables (rbf_cutoff,
    n_gaussians, variance) are taken as suggested in SchnetPack.

    In practice, we have observed that ShiftedSoftplus as an activation
    function for a SchnetFeature that is used for a CGnet will lead
    to simulation instabilities when using that CGnet to generate new
    data. We have experienced more success with nn.Tanh().

    Example
    -------
    # Basic example on how to initialize and use the SchnetFeature class.
    # First, initialize an embedding.
    embedding_size = 5
    embedding_layer = CGBeadEmbedding(n_embeddings=5,
                                      embedding_dim=10)

    beads = 5  # example number of coarse-grain beads in the dataset

    # Next, create a SimpleNormLayer to normalize ContinuousFilterConvolution
    # output.
    norm_layer = SimpleNormLayer(n_beads)

    schnet_feature = SchnetFeature(feature_size=10,
                                   embedding_layer=embedding_layer,
                                   n_interaction_blocks=2,
                                   calculate_geometry=True,
                                   n_beads=beads,
                                   normalization_layer=norm_layer,
                                   neighbor_cutoff=5.0)

    # To perform a forward pass, pass coordinates and properties to embed to
    # schnet_feature.
    # Properties should be integers, e.g. nuclear charge.

    # coordinates is a torch.Tensor size [n_frames, n_beads, 3]
    # embedding_properties is a torch.Tensor size [n_frames, n_beads]
    schnet_features = schnet_feature(coordinates, embedding_properties)

    # In case SchnetFeature is initialized with calculate_geometry=False,
    # distances instead of coordinates can passed.
    # Distances should have the size [n_frames, n_beads, n_beads-1].

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
    """

    def __init__(self,
                 feature_size,
                 embedding_layer,
                 rbf_layer,
                 n_beads,
                 activation=ShiftedSoftplus(),
                 calculate_geometry=None,
                 neighbor_cutoff=None,
                 normalization_layer=None,
                 n_interaction_blocks=1,
                 share_weights=False,
                 share_batchnorm_parameters=False,
                 device=torch.device('cpu')):
        super(SchnetFeature, self).__init__()
        self.device = device
        self.geometry = Geometry(method='torch', device=self.device)
        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer
        basis_size = len(rbf_layer)

        if share_weights:
            # Lets the interaction blocks share weight parameters.
            # If batchnorm layers are used, their parameters
            # are shared as well.
            if share_batchnorm_parameters:
                # Batchnorm parameters are the same for all instances
                self.interaction_blocks = nn.ModuleList(
                    [InteractionBlock(feature_size, basis_size,
                                      feature_size, activation=activation,
                                      normalization_layer=normalization_layer)]
                    * n_interaction_blocks
                )
            if not share_batchnorm_parameters:
                # This represents the case where weights parameters are 
                # shared between the interaction blocks, but the batchnorm
                # parameters are not shared.
                self.interaction_blocks = nn.ModuleList(
                    [InteractionBlock(feature_size, basis_size,
                                      feature_size, activation=activation,
                                      normalization_layer=normalization_layer)]
                    * n_interaction_blocks
                )
                # We reinstance each bathcnorm layer in each interaction block
                # as a deep copy of the original layer in order to prevent parameter
                # sharing between the batchnorm layers
                for interaction_block in self.interaction_blocks:
                    if isinstance(normalization_layer, nn.BatchNorm1d):
                        # get batchnorm parameters to make deep copies
                        # that do not have linked parameters
                        num_features = normalization.num_features
                        eps = normalization.eps
                        momentum = normalization.momentum
                        affine = normalization.affine
                        track_running_stats = normalization.track_running_stats
                        new_layer = nn.BatchNorm1d(num_features, eps=eps,
                                                   momentum=momentum,
                                                   affine=affine,
                                                   track_running_stats=track_running_stats)
                        interaction_block.cfconv.normlayer = new_layer

        else:
            # Every interaction block has their own weights
            self.interaction_blocks = nn.ModuleList(
                [InteractionBlock(feature_size, basis_size,
                                  feature_size, activation=activation,
                                  normalization_layer=normalization_layer)
                 for _ in range(n_interaction_blocks)]
            )

        self.neighbor_cutoff = neighbor_cutoff
        self.calculate_geometry = calculate_geometry
        if self.calculate_geometry:
            pass
        else:
            self._distance_pairs, _ = self.geometry.get_distance_indices(n_beads,
                                                                         [], [])
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

        Notes
        -----
        If a dataset with variable molecule sizes is being used, it is
        important to mask the contributions from padded portions of
        the input into the neural network. This is done using the batchwise
        variables 'bead_mask' (shape [n_frames, n_beads]) and
        'bead_distance_mask' (shape [n_frames, n_beads, n_neighbors]).
        These masks are used to set contributions from non-physical beads
        to zero through elementwise multiplication with layer outputs
        """

        # if geometry is specified, the distances are calculated from input
        # coordinates. Otherwise, it is assumed that in_features are
        # pairwise distances in redundant form
        if self.calculate_geometry:
            n_beads = embedding_property.size()[1]
            self._distance_pairs, _ = self.geometry.get_distance_indices(n_beads,
                                                                         [], [])
            redundant_distance_mapping = self.geometry.get_redundant_distance_mapping(
                self._distance_pairs)
            distances = self.geometry.get_distances(self._distance_pairs,
                                                    in_features, norm=True)
            distances = distances[:, redundant_distance_mapping]
        else:
            distances = in_features

        neighbors, neighbor_mask = self.geometry.get_neighbors(distances,
                                                               cutoff=self.neighbor_cutoff)

        neighbors = neighbors.to(self.device)
        neighbor_mask = neighbor_mask.to(self.device)
        batches, _, neighbor_beads = neighbors.size()

        # Here we make the beadwise mask from the embedding property information
        # for use in masking out non-physical beads introduced by variable
        # length padding. The embeddings of these non-physical beads are all
        # 0s, as constructed by collating function if multi_molecule_collate
        # is used. We can directly make a mask from this variable by creating
        # a copy in which the values are clamped to be either 0 or 1.
        # This mask plays many important roles, such as filtering out
        # non-physical distances, energies, and other properties propgated
        # through the network that might ultimately add non-physical contributions
        # to the loss function used during training
        bead_mask = torch.clamp(embedding_property, min=0, max=1).float()

        # A similar mask is made to mask the distances in redundant form
        # so that they do not contain distances evaluated using non-physical
        # atoms introduced by padding.
        # This mask has shape [n_frames, max_n_beads, max_n_neighbors]
        bead_distances_mask = (bead_mask[:,:,None].repeat(1,1,neighbor_beads)*bead_mask[None,:,1:].view(batches,1,neighbor_beads)).float()

        # In order to prevent backpropagation instabilities associated with
        # evaluating square root derivatives at 0, the masking must be done
        # in the following way. This method also avoids in-place operations
        # that are not compatible with backward gradient flow
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[bead_distances_mask != 0] = distances[bead_distances_mask != 0]
        distances = tmp_distances * neighbor_mask

        features = self.embedding_layer(embedding_property)
        rbf_expansion = self.rbf_layer(distances=distances,
                                       distance_mask=bead_distances_mask)

        for interaction_block in self.interaction_blocks:
            interaction_features = interaction_block(features=features,
                                                     rbf_expansion=rbf_expansion,
                                                     neighbor_list=neighbors,
                                                     neighbor_mask=neighbor_mask,
                                                     bead_mask=bead_mask)
            features = features + interaction_features

        return features
