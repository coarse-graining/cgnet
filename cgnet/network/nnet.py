# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang

import torch
import torch.nn as nn
import numpy as np
from .priors import ZscoreLayer, HarmonicLayer, RepulsionLayer
from cgnet.feature import FeatureCombiner, SchnetFeature, GeometryFeature


class ForceLoss(torch.nn.Module):
    """Loss function for force matching scheme."""

    def __init__(self):
        super(ForceLoss, self).__init__()

    def forward(self, force, labels):
        """Returns force matching loss averaged over all examples.

        Parameters
        ----------
        force : torch.Tensor (grad enabled)
            forces calculated from the CGnet energy via autograd.
            Size [n_frames, n_degrees_freedom].
        labels : torch.Tensor
            forces to compute the loss against. Size [n_frames,
                                                      n_degrees_of_freedom].

        Returns
        -------
        loss : torch.Variable
            example-averaged Frobenius loss from force matching. Size [1, 1].

        """
        loss = ((force - labels)**2).mean()
        return loss


class CGnet(nn.Module):
    """CGnet neural network class

    Parameters
    ----------
    arch : list of nn.Module() instances
        underlying sequential network architecture.
    criterion : nn.Module() instances
        loss function to be used for network.
    feature : nn.Module() instance
        feature layer to transform cartesian coordinates into roto-
        translationally invariant features.
    priors : list of nn.Module() instances (default=None)
        list of prior layers that provide energy contributions external to
        the hidden architecture of the CGnet.

    Notes
    -----
    CGnets are a class of feedforward neural networks introduced by Jiang et.
    al. (2019) which are used to predict coarse-grained molecular force fields
    from Cartesain coordinate data. They are characterized by an autograd layer
    with respect to input coordinates implemented before the loss function,
    which directs the network to learn a representation of the coarse-grained
    potential of mean force (PMF) associated with a conservative coarse-grained
    force feild via a gradient operation as prescribed by classical mechanics.
    CGnets may also contain featurization layers, which transform Cartesian
    inputs into roto-translationally invariant features, thereby yeilding a PMF
    that respects these invarainces. CGnets may additionally be supplied with
    external prior functions, which are useful for regularizing network behavior
    in sparsely sampled, unphysical regions of molecular configuration space.

    The inputs to CGnet (coordinates and forces) determine the units
    that are learned/used by CGnet. It is important to make sure that the units
    between the input coordinates and force labels are consistent with one
    another. These units must also be consistent with the interaction
    parameters for and specified priors. If one desires to use CGnet to make
    predictions in a different unit system, the predictions must be made using
    original unit system, and then converted to the desired unit system
    outside of the CGnet. Otherwise, a new CGnet model must be trained using the
    desired units. 

    Examples
    --------
    From Jiang et. al. (2019), the optimal architecture for a 5-bead coarse
    grain model of alanine dipeptide, featurized into bonds, angles, pairwaise
    distances, and backbone torsions, was found to be:

    CGnet(
      (input): in_features=30
      (arch): Sequential(
        (0): GeometryFeature(in_features=30, out_features=17)
        (1): Linear(in_features=17, out_features=160, bias=True)
        (2): Tanh()
        (3): Linear(in_features=160, out_features=160, bias=True)
        (4): Tanh()
        (5): Linear(in_features=160, out_features=160, bias=True)
        (6): Tanh()
        (7): Linear(in_features=160, out_features=160, bias=True)
        (8): Tanh()
        (9): Linear(in_features=160, out_features=160, bias=True)
        (10): Tanh()
        (11): Linear(in_features=160, out_features=1, bias=True)
        (12): HarmonicLayer(bonds)
        (13): HarmonicLayer(angles)
        (14): torch.autograd.grad(-((11) + (12) + (13)), input,
                                  create_graph=True, retain_graph=True)
      )
    (criterion): ForceLoss()
    )

    Mounting to GPU can be accomplished using the 'mount' method. For example,
    given an instance of CGnet and a torch.device, the model may be mounted in
    the follwing way:

       my_cuda = torch.device('cuda')
       model.mount(my_cuda)

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E.,
        de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913

    """

    def __init__(self, arch, criterion, feature=None, priors=None):
        super(CGnet, self).__init__()
        zscore_idx = 1
        for layer in arch:
            if isinstance(layer, ZscoreLayer):
                self.register_buffer('zscores_{}'.format(zscore_idx),
                                     layer.zscores)

                zscore_idx += 1
        self.arch = nn.Sequential(*arch)
        if priors:
            self.priors = nn.Sequential(*priors)
            harm_idx = 1
            repul_idx = 1
            for layer in self.priors:
                if isinstance(layer, HarmonicLayer):
                    self.register_buffer('harmonic_params_{}'.format(harm_idx),
                                         layer.harmonic_parameters)
                    harm_idx += 1
                if isinstance(layer, RepulsionLayer):
                    self.register_buffer('repulsion_params_{}'.format(repul_idx),
                                         layer.repulsion_parameters)
                    repul_idx += 1
        else:
            self.priors = None
        self.criterion = criterion
        self.feature = feature

    def forward(self, coordinates, embedding_property=None):
        """Forward pass through the network ending with autograd layer.

        Parameters
        ----------
        coord : torch.Tensor (grad enabled)
            input trajectory/data of size [n_frames, n_degrees_of_freedom].
        embedding_property: torch.Tensor (default=None)
            Some property that should be embedded. Can be nuclear charge
            or maybe an arbitrary number assigned for amino-acids.
            Size [n_frames, n_properties]

        Returns
        -------
        energy : torch.Tensor
            scalar potential energy of size [n_frames, 1]. If priors are
            supplied to the CGnet, then this energy is the sum of network
            and prior energies.
        force  : torch.Tensor
            vector forces of size [n_frames, n_degrees_of_freedom].

        Notes
        -----
        If a dataset with variable molecule sizes is being used, it is
        important to mask the contributions from padded portions of
        the input into the neural network. This is done using the batchwise
        variable 'bead_mask' (shape [n_frames, n_beads]).
        This mask is used to set energy contributions from non-physical beads
        to zero through elementwise multiplication with the CGnet ouput for
        models using SchnetFeatures
        """

        if self.feature:
            # The below code adheres to the following logic:
            # 1. The feature_output is always what is passed to the model architecture
            # 2. The geom_feature is always what is passed to the priors
            # There will never be no feature_output, but sometimes it will
            # be the same as the geom_feature. There may be no geom_feature.
            if isinstance(self.feature, FeatureCombiner):
                # Right now, the only case we have is that a FeatureCombiner
                # with two Features will be a SchnetFeature followed by a
                # GeometryFeature.
                feature_output, geom_feature = self.feature(coordinates,
                                                            embedding_property=embedding_property)
                if self.feature.propagate_geometry:
                    # We only can use propagate_geometry if the feature_output is a 
                    # SchnetFeature
                    schnet_feature = feature_output
                    if geom_feature is None:
                        raise RuntimeError(
                            "There is no GeometryFeature to propagate. Was " \
                            "your FeatureCombiner a SchnetFeature only?"
                            )
                    n_frames = coordinates.shape[0]
                    schnet_feature = schnet_feature.reshape(n_frames, -1)
                    concatenated_feature = torch.cat((schnet_feature, geom_feature), dim=1)
                    energy = self.arch(concatenated_feature)
                else:
                    energy = self.arch(feature_output)
                if len(energy.size()) == 3:
                    # sum energy over beads
                    energy = torch.sum(energy, axis=1)
            if not isinstance(self.feature, FeatureCombiner):
                if embedding_property is not None:
                    # This assumes the only feature with an embedding_property
                    # is a SchnetFeature. If other features can take embeddings,
                    # this needs to be revisited.
                    feature_output = self.feature(
                        coordinates, embedding_property)
                    geom_feature = None
                else:
                    feature_output = self.feature(coordinates)
                    geom_feature = feature_output
                energy = self.arch(feature_output)
        else:
            # Finally, if we pass only the coordinates with no pre-computed
            # Feature, then we call those coordinates the feature. We will
            # name this geom_feature because there may be priors on it.
            feature_output = coordinates
            geom_feature = coordinates
            energy = self.arch(feature_output)
        if self.priors:
            if geom_feature is None:
                raise RuntimeError(
                    "Priors may only be used with GeometryFeatures or coordinates."
                )
            for prior in self.priors:
                if isinstance(prior, _EmbeddingPriorLayer):
                    energy = energy + prior(geom_feature[:, prior.callback_indices,
                                            embedding_property)
                else:
                    energy = energy + prior(geom_feature[:, prior.callback_indices])
        # Sum up energies along bead axis for Schnet outputs and mask out
        # nonexisting beads
        if len(energy.size()) == 3 and isinstance(self.feature, SchnetFeature):
            # Make sure to mask those beads which are not physical.
            # Their contribution to the predicted energy and forces
            # should be zero
            bead_mask = torch.clamp(embedding_property, min=0, max=1).float()
            masked_energy = energy * bead_mask[:, :, None]
            energy = torch.sum(masked_energy, axis=-2)
        # Perform autograd to learn potential of conservative force field
        force = torch.autograd.grad(-torch.sum(energy),
                                    coordinates,
                                    create_graph=True,
                                    retain_graph=True)
        return energy, force[0]

    def mount(self, device):
        """Wrapper for device mounting

        Parameters
        ----------
        device : torch.device
            Device upon which model can be mounted for computation/training
        """

        # Buffers and parameters
        self.to(device)
        # Non parameters/buffers
        if self.feature:
            if isinstance(self.feature, FeatureCombiner):
                for layer in self.feature.layer_list:
                    if isinstance(layer, (GeometryFeature, SchnetFeature)):
                        layer.device = device
                        layer.geometry.device = device
                    if isinstance(layer, ZscoreLayer):
                        layer.to(device)
            if isinstance(self.feature, (GeometryFeature, SchnetFeature)):
                self.feature.device = device
                self.feature.geometry.device = device

    def predict(self, coord, force_labels, embedding_property=None):
        """Prediction over test/validation batch.

        Parameters
        ----------
        coord: torch.Tensor (grad enabled)
            input trajectory/data of size [n_frames, n_degrees_of_freedom]
        force_labels: torch.Tensor
            force labels of size [n_frames, n_degrees_of_freedom]
        embedding_property: torch.Tensor (default=None)
            Some property that should be embedded. Can be nuclear charge
            or maybe an arbitrary number assigned for amino-acids.
            Size [n_frames, n_properties]
        Returns
        -------
        loss.data : torch.Tensor
            loss over prediction inputs.

        """

        self.eval()  # set model to eval mode
        energy, force = self.forward(coord)
        loss = self.criterion.forward(force, force_labels,
                                      embedding_property=embedding_property)
        self.train()  # set model to train mode
        return loss.data
