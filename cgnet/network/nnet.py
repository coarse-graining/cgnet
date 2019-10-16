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
        """
        if self.feature:
            if isinstance(self.feature, FeatureCombiner):
                forward_feat, feat = self.feature(coordinates,
                                                  embedding_property=embedding_property)
                energy = self.arch(forward_feat)
                if len(energy.size()) == 3:
                    # sum energy over beads
                    energy = torch.sum(energy, axis=1)
            if not isinstance(self.feature, FeatureCombiner):
                if embedding_property is not None:
                    feat = self.feature(coordinates, embedding_property)
                else:
                    feat = self.feature(coordinates)
                energy = self.arch(feat)
        else:
            feat = coordinates
            energy = self.arch(feat)
        if self.priors:
            for prior in self.priors:
                energy = energy + prior(feat[:, prior.callback_indices])
        # Sum up energies along bead axis for Schnet outputs
        if len(energy.size()) == 3 and isinstance(self.feature, SchnetFeature):
            energy = torch.sum(energy, axis=-2)
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
