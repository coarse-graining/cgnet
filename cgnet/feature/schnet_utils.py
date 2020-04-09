# Author: Dominik Lemm
# Contributors: Nick Charron, Brooke Husic

import torch
import torch.nn as nn

from cgnet.feature.utils import ShiftedSoftplus, LinearLayer


def _check_normalization_input(n_beads):
    """Helper function for ensuring n_beads is an
    integer greater than or equal to one. This helper function is used
    by both ContinuousFilterConvolution and InteractionBlock to
    ensure that the the instancing of either a simple bead number
    normalizaiton or beadwise batch normalization after the continuous
    convolution filter output is done with a proper integer that
    represents the number of beads in the system.

    Parameters
    ----------
    n_beads: any type
        the input value used to construct either an nn.BatchNorm1d instance
        within a ContinuousFilterConvolution, or a simple bead number
        normalization after element-wise filter convolution.
    """

    # Make sure n_beads is an integer
    if not isinstance(n_beads, int):
        raise ValueError(
            "n_beads must be an integer.")
    else:
        # Make sure n_beads is specifically not a bool (Python 3.7 types
        # bools as integers)
        if isinstance(n_beads, bool):
            raise ValueError(
                "n_beads must be specified by an integer, not a bool.")
        # Make sure n_beads, if an integer, is greater than or equal to one
        if n_beads < 1:
            raise ValueError(
                "n_beads must be positive.")


class CGBeadEmbedding(torch.nn.Module):
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

    def __init__(self, n_embeddings, embedding_dim):
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
            in the case of zero padded batches). The properties to be embedded
            should be integers (torch type long).
            Size [n_frames, n_beads]

        Returns
        -------
        embedding_vector: torch.Tensor
            Corresponding embedding vector to the passed indices.
            Size [n_frames, n_beads, embedding_dim]
        """
        return self.embedding(embedding_property)


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
    activation: nn.Module (default=ShiftedSoftplus())
        Activation function for the filter generating network. Following
        Schütt et al, the default value is ShiftedSoftplus, but any
        differentiable activation function can be used (see Notes).
    simple_norm: int (default=None)
        Number of beads in the system, with which the output of the continuous
        filter convolution will be normalized
    beadwise_batchnorm: int (default=None)
        Number of beads over which batch normalization will be applied after
        application of the continuous filter convolution. If None, batch
        normalization will not be used
    batchnorm_running_stats: bool (default=False)
        If beadwise_batchnorm is not None, this argument populates the
        track_running_stats argument in torch.nn.BatchNorm1d

    Notes
    -----
    Following the current implementation in SchNetPack, the last linear layer of
    the filter generator does not contain an activation function.
    This allows the filter generator to contain negative values.

    In practice, we have observed that ShiftedSoftplus as an activation
    function for a SchnetFeature (i.e., within its ContinuousFilterConvolution)
    that is used for a CGnet will lead to simulation instabilities when using
    that CGnet to generate new data. We have experienced more success with
    nn.Tanh().

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
    """

    def __init__(self, n_gaussians, n_filters, activation=ShiftedSoftplus(),
                 simple_norm=None, beadwise_batchnorm=None,
                 batchnorm_running_stats=False):
        super(ContinuousFilterConvolution, self).__init__()
        filter_layers = LinearLayer(n_gaussians, n_filters, bias=True,
                                    activation=activation)
        # No activation function in the last layer allows the filter generator
        # to contain negative values.
        filter_layers += LinearLayer(n_filters, n_filters, bias=True)
        self.filter_generator = nn.Sequential(*filter_layers)

        if beadwise_batchnorm and simple_norm:
            raise RuntimeError('beadwise_batchnorm and simple_norm '
                               'cannot be used simultaneously')
        elif beadwise_batchnorm != None:
            _check_normalization_input(beadwise_batchnorm)
            self.normlayer = nn.BatchNorm1d(beadwise_batchnorm,
                                            track_running_stats=batchnorm_running_stats)
        elif simple_norm != None:
            _check_normalization_input(simple_norm)
            self.normlayer = simple_norm
        elif beadwise_batchnorm == None and simple_norm == None:
            self.normlayer = None

    def forward(self, features, rbf_expansion, neighbor_list, neighbor_mask, bead_mask=None):
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
        neighbor_mask: torch.Tensor
            Index mask to filter out non-existing neighbors that were
            introduced to due distance cutoffs or padding.
            Size [n_frames, n_beads, n_neighbors]
        bead_mask: torch.Tensor (default=None)
            Mask used to filter out non-existing beads that may be
            present in datasets with molecules of different sizes
            Size [n_frames, n_beads, n_neighbors]

        Returns
        -------
        aggregated_features: torch.Tensor
            Residual features of shape [n_frames, n_beads, n_features]

        """

        # Generate the convolutional filter
        # Size (n_frames, n_beads, n_neighbors, n_features)
        conv_filter = self.filter_generator(rbf_expansion)

        # Feature tensor needs to be transformed from
        # (n_frames, n_beads, n_features)
        # to
        # (n_frames, n_beads, n_neighbors, n_features)
        # This can be done by feeding the features of a respective bead into
        # its position in the neighbor_list.
        n_batch, n_beads, n_neighbors = neighbor_list.size()

        # Size (n_frames, n_beads * n_neighbors, 1)
        neighbor_list = neighbor_list.reshape(-1, n_beads * n_neighbors, 1)
        # Size (n_frames, n_beads * n_neighbors, n_features)
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

        # Remove features from non-existing neighbors outside the cutoff
        conv_features = conv_features * neighbor_mask[:, :, :, None]
        ##print("Masked conv_features:")
        ##print(conv_features)
        # Aggregate/pool the features from (n_frames, n_beads, n_neighs, n_feats)
        # to (n_frames, n_beads, n_features)
        aggregated_features = torch.sum(conv_features, dim=2)

        # Filter out contributions from non-existent beads introduced by padding
        #print(aggregated_features.size())
        #print(bead_mask.size())
        if bead_mask is not None:
            aggregated_features = aggregated_features * bead_mask[:,:,None]

        # TODO This needs to be changed to normalize by neighbor number not
        # bead number

        if self.normlayer is not None:
            if isinstance(self.normlayer, nn.BatchNorm1d):
                return self.normlayer(aggregated_features)
            elif isinstance(self.normlayer, int):
                return aggregated_features / self.normlayer
        else:
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
    activation: nn.Module (default=ShiftedSoftplus())
        Activation function for the atom-wise layers. Following Schütt et al,
        the default value is ShiftedSoftplus, but any differentiable activation
        function can be used (see Notes).
    simple_norm: int (default=None)
        Number of beads in the system, with which the output of the continuous
        filter convolution will be normalized
    beadwise_batchnorm: int (default=None)
        Number of beads over which batch normalization will be applied after
        application of the continuous filter convolution. If None, batch
        normalization will not be used
    batchnorm_running_stats: bool (default=False)
        If beadwise_batchnorm is not None, this argument populates the
        track_running_stats argument in torch.nn.BatchNorm1d

    Notes
    -----
    The additive residual connection between interaction blocks is not
    included in the output of this forward pass. The residual connection
    will be computed separately outside of this class.

    In practice, we have observed that ShiftedSoftplus as an activation
    function for a SchnetFeature (i.e., within its InteractionBlock)
    that is used for a CGnet will lead to simulation instabilities when using
    that CGnet to generate new data. We have experienced more success with
    nn.Tanh().

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
    """

    def __init__(self, n_inputs, n_gaussians, n_filters,
                 activation=ShiftedSoftplus(), simple_norm=None,
                 beadwise_batchnorm=None, batchnorm_running_stats=False):
        super(InteractionBlock, self).__init__()

        self.initial_dense = nn.Sequential(
            *LinearLayer(n_inputs, n_filters, bias=False,
                         activation=None))
        # backwards compatibility for spelling error in initial dense
        # layer attribute.
        # WARNING : This will be removed in the future!
        self.inital_dense = self.initial_dense

        if beadwise_batchnorm and simple_norm:
            raise RuntimeError('beadwise_batchnorm and simple_norm '
                               'cannot be used simultaneously')
        else:
            if beadwise_batchnorm != None:
                _check_normalization_input(beadwise_batchnorm)
            if simple_norm != None:
                _check_normalization_input(simple_norm)
        self.cfconv = ContinuousFilterConvolution(n_gaussians=n_gaussians,
                                                  n_filters=n_filters,
                                                  activation=activation,
                                                  simple_norm=simple_norm,
                                                  beadwise_batchnorm=beadwise_batchnorm,
                                                  batchnorm_running_stats=batchnorm_running_stats)
        output_layers = LinearLayer(n_filters, n_filters, bias=True,
                                    activation=activation)
        output_layers += LinearLayer(n_filters, n_filters, bias=True,
                                     activation=None)
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, features, rbf_expansion, neighbor_list, neighbor_mask, bead_mask=None):
        """ Compute interaction block

        Parameters
        ----------
        features: torch.Tensor
            Input features from an embedding or interaction layer.
            Size [n_frames, n_beads, n_features]
        rbf_expansion: torch.Tensor
            Radial basis function expansion of inter-bead distances.
            Size [n_frames, n_beads, n_neighbors, n_gaussians]
        neighbor_list: torch.Tensor
            Indices of all neighbors of each bead.
            Size [n_frames, n_beads, n_neighbors]
        neighbor_mask: torch.Tensor
            Index mask to filter out non-existing neighbors that were
            introduced to due distance cutoffs.
            Size [n_frames, n_beads, n_neighbors]
        bead_mask: torch.Tensor (default=None)
            Mask used to filter out non-existing beads that may be
            present in datasets with molecules of different sizes
            Size [n_frames, n_beads, n_neighbors]

        Returns
        -------
        output_features: torch.Tensor
            Output of an interaction block. This output can be used to form
            a residual connection with the output of a prior embedding/interaction
            layer.
            Size [n_frames, n_beads, n_filters]

        """
        init_feature_output = self.initial_dense(features)
        conv_output = self.cfconv(init_feature_output, rbf_expansion,
                                  neighbor_list, neighbor_mask, bead_mask=bead_mask)
        output_features = self.output_dense(conv_output)
        return output_features
