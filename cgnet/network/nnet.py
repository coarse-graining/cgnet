# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang

import torch
import torch.nn as nn
import numpy as np


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
            Size [n_examples, n_degrees_freedom].
        labels : torch.Tensor
            forces to compute the loss against. Size [n_examples,
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

    References
    ----------
    Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E.,
        de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning
        of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science.
        https://doi.org/10.1021/acscentsci.8b00913

    """

    def __init__(self, arch, criterion, feature=None, priors=None):
        super(CGnet, self).__init__()

        self.arch = nn.Sequential(*arch)
        if priors:
            self.priors = nn.Sequential(*priors)
        else:
            self.priors = None
        self.criterion = criterion
        self.feature = feature

    def forward(self, coord):
        """Forward pass through the network ending with autograd layer.

        Parameters
        ----------
        coord : torch.Tensor (grad enabled)
            input trajectory/data of size [n_examples, n_degrees_of_freedom].

        Returns
        -------
        energy : torch.Tensor
            scalar potential energy of size [n_examples, 1]. If priors are
            supplied to the CGnet, then this energy is the sum of network
            and prior energies.
        force  : torch.Tensor
            vector forces of size [n_examples, n_degrees_of_freedom].
        """
        feat = coord
        if self.feature:
            feat = self.feature(feat)

        # forward pass through the hidden architecture of the CGnet
        energy = self.arch(feat)
        # addition of external priors to form total energy
        if self.priors:
            for prior in self.priors:
                energy = energy + prior(feat[:, prior.callback_indices])

        # Perform autograd to learn potential of conservative force field
        force = torch.autograd.grad(-torch.sum(energy),
                                    coord,
                                    create_graph=True,
                                    retain_graph=True)
        return energy, force[0]

    def predict(self, coord, force_labels):
        """Prediction over test/validation batch.

        Parameters
        ----------
        coord: torch.Tensor (grad enabled)
            input trajectory/data of size [n_examples, n_degrees_of_freedom]
        force_labels: torch.Tensor
            force labels of size [n_examples, n_degrees_of_freedom]

        Returns
        -------
        loss.data : torch.Tensor
            loss over prediction inputs.

        """

        self.eval()  # set model to eval mode
        energy, force = self.forward(coord)
        loss = self.criterion.forward(force, force_labels)
        self.train()  # set model to train mode
        return loss.data
