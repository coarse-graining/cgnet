# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

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
            forces to compute the loss against. Size [n_examples,n_degrees_of_freedom].

        Returns
        -------
        loss : torch.Variable
            example-averaged Frobenius loss from force matching. Size [1, 1].

        """
        loss = ((force - labels)**2).mean()
        return loss


def LinearLayer(d_in, d_out, bias=True, activation=None, dropout=0, weight_init='xavier',
                weight_init_args=None, weight_init_kwargs=None):
    """Linear layer function

    Parameters
    ----------
    d_in : int
        input dimension
    d_out : int
        output dimension
    bias : bool (default=True)
        specifies whether or not to add a bias node
    activation : torch.nn.Module() (default=None)
        activation function for the layer
    dropout : float (default=0)
        if > 0, a dropout layer with the specified dropout frequency is
        added after the activation.
    weight_init : str, float, or nn.init function (default=\'xavier\')
        specifies the initialization of the layer weights. For non-option
        initializations (eg, xavier initialization), a string may be used
        for simplicity. If a float or int is passed, a constant initialization
        is used. For more complicated initializations, a torch.nn.init function
        object can be passed in.
    weight_init_args : list or tuple (default=None)
        arguments (excluding the layer.weight argument) for a torch.nn.init
        function.
    weight_init_kwargs : dict (default=None)
        keyword arguements for a torch.nn.init function

    Returns
    -------
    seq : list of torch.nn.Module() instances
        the full linear layer, including activation and optional dropout.

    Example
    -------
    MyLayer = LinearLayer(5,10,bias=True,activation=nn.Softplus(beta=2),
                               weight_init=nn.init.kaiming_uniform_,
                               weight_init_kwargs={"a":0,"mode":"fan_out",
                               "nonlinearity":"leaky_relu"})

    Produces a linear layer with input dimension 5, output dimension 10, bias
    inclusive, followed by a beta=2 softplus activation, with the layer weights
    intialized according to kaiming uniform procedure with preservation of weight
    variance magnitudes during backpropagation.

    """

    seq = [nn.Linear(d_in, d_out, bias=bias)]
    if activation:
        seq += [activation]
    if dropout:
        seq += [nn.Dropout(dropout)]
    if weight_init == 'xavier':
        torch.nn.init.xavier_uniform_(seq[0].weight)
    if weight_init == 'identity':
        torch.nn.init.eye_(seq[0].weight)
    if isinstance(weight_init, int) or isinstance(weight_init, float):
        torch.nn.init.constant_(seq[0].weight, weight_init)
    if callable(weight_init):
        if weight_init_args is None:
            weight_init_args = []
        if weight_init_kwargs is None:
            weight_inti_kwargs = []
        weight_init(seq[0].weight, *weight_init_args, **weight_init_kwargs)
    return seq

class Net(nn.Module):
    """CGnet neural network class

    Parameters
    ----------
    arch : list of nn.Module() instances
        underlying sequential network architecture.
    criterion : nn.Module() instances
        loss function to be used for network.
    """

    def __init__(self, arch, criterion):
        super(Net, self).__init__()

        self.arch = nn.Sequential(*arch)
        self.criterion = criterion

    def forward(self, coord):
        """Forward pass through the network

        Parameters
        ----------
        coord : torch.Tensor (grad enabled)
            input trajectory/data of size [n_examples, n_degrees_of_freedom].

        Returns
        -------
        energy : torch.Tensor
            scalar potential energy of size [n_examples, 1].
        force  : torch.Tensor
            vector forces of size [n_examples, n_degrees_of_freedom].
        """

        energy = self.arch(coord)
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
