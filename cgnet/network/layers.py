# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang

import torch
import torch.nn as nn
import numpy as np


class _PriorLayer(nn.Module):
    """Layer for adding prior energy computations external to CGnet hidden
    output

    Parameters
    ----------
    feat_data: dict
        dictionary defining the CG beads and interaction parameters for
        computing the energy contributions of the residual prior energy. The
        keys are tuples defining the CG beads involved in each interaction,
        and the values are dictionaries of physical constants names/values
        (keys: strings, values: float) involved in the interaction encompassed
        by those CG beads
    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys. Possible
        feature keys are those implemented in ProteinBackBoneStatistics():
        \"Distacnces\", \"Angles\", \"Dihedral_cosines\", and/or
        \"Dihedral_sines\"
    feature_type: str
        features type from which to select coordinates.
    """

    def __init__(self, feat_data, descriptions=None, feature_type=None):
        super(_PriorLayer, self).__init__()
        if descriptions and not feature_type:
            raise RuntimeError('Must declare feature_type if using \
                                descriptions')
        if feature_type not in descriptions.keys():
            raise ValueError('Feature type not found in descrptions')
        if descriptions and feature_type:
            self.params = []
            self.feature_type = feature_type
            self.features = [feat for feat in feat_data.keys()]
            self.feat_idx = []
            # get number of each feature to determine starting idx
            nums = [len(descriptions['Distances']), len(descriptions['Angles']),
                    len(descriptions['Dihedral_cosines']),
                    len(descriptions['Dihedral_sines'])]
            descs = ['Distances', 'Angles',
                     'Dihedral_cosines', 'Dihedral_sines']
            self.start_idx = 0
            for num, desc in zip(nums, descs):
                if self.feature_type == desc:
                    break
                else:
                    self.start_idx += num
            for key, par in feat_data.items():
                self.features.append(key)
                self.feat_idx.append(self.start_idx +
                                     descriptions[self.feature_type].index(key))
                self.params.append(par)

    def forward(self, in_feat):
        """Forward method to compute the prior energy contribution.

        Notes
        -----
        This must be explicitly implemented in a child class that inherits from
        _PriorLayer(). The details of this method should encompass the
        mathematical steps to form each specific energy contribution to the
        potential energy.;
        """
        raise NotImplementedError


class RepulsionLayer(_PriorLayer):
    """Layer for calculating pairwise repulsion energy prior.

    Parameters
    ----------
    feat_data: dict
        dictionary defining the CG beads and interaction parameters for
        computing the energy contributions of the residual prior energy. The
        keys are tuples defining the CG beads involved in each pairwise
        interaction, and the values are dictionaries of physical constants
        involved in the corresponding repulsion interaction: The keys of this
        subdictionary are \"ex_vol\", and \"exp\", which are the exlcuded volume
        parameter (in length units) and the exponent (positive, dimensionless)
        respectively. The corresponding values are the the numerical values of
        each constant. For example, for one such feat_dict entry:

            { (3, 9) : {  \"ex_vol\" : 5.5, \"exp\" : 6.0 }}

    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys. Possible
        feature keys are those implemented in ProteinBackBoneStatistics():
        \"Distacnces\", \"Angles\", \"Dihedral_cosines\", and/or
        \"Dihedral_sines\"
    feature_type: str
        features type from which to select coordinates.

    Notes
    -----
    This prior energy should be used for longer molecules that may possess
    metastable states in which portions of the molecule that are separated by
    many CG beads in sequence may nonetheless adopt close physical proximities.
    Without this prior, it is possilbe for the CGnet to learn energies that do
    not respect proper physical pairwise repulsions. The interaction is modeled
    after the VDW interaction term from the classic Leonard Jones potential.

    """

    def __init__(self, feat_data, descriptions=None, feature_type=None):
        super(RepulsionLayer, self).__init__(feat_data,
                                             descriptions=descriptions,
                                             feature_type=feature_type)
        self.repulsion_parameters = torch.tensor([])
        for param_dict in self.params:
            self.repulsion_parameters = torch.cat((self.repulsion_parameters,
                                        torch.tensor([[param_dict['ex_vol']],
                                        [param_dict['exp']]])), dim=1)

    def forward(self, in_feat):
        """Calculates repulsion interaction contributions to energy
        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as pairwise distances, of size (n,k), for
            n examples and k features.
        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.
        """

        n = len(in_feat)
        energy = torch.sum((self.repulsion_parameters[0, :]/in_feat)
                           ** self.repulsion_parameters[1, :],
                           1).reshape(n, 1) / 2
        return energy


class HarmonicLayer(_PriorLayer):
    """Layer for calculating bond/angle harmonic energy prior

    Parameters
    ----------
    feat_data: dict
        dictionary of means and bond constants. Keys are tuples that provide the
        descriptions of each contributing feature. THe values of each key are in
        turn dictionarys that have the following keys. The \'mean\' key is
        mapped to the numerical mean of the feature over the trajectory. The
        \'std\' key is mapped to the numerical standard deviation of the feature
        over the trajectory. The \'k\' is mapped to the harmonic constant
        derived from the feature.
    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys.
    feature_type: str
        features type from which to select coordinates.

    """

    def __init__(self, feat_data, descriptions=None, feature_type=None):
        super(HarmonicLayer, self).__init__(feat_data,
                                            descriptions=descriptions,
                                            feature_type=feature_type)
        self.harmonic_parameters = torch.tensor([])
        for param_dict in self.params:
            self.harmonic_parameters = torch.cat((self.harmonic_parameters,
                                       torch.tensor([[param_dict['k']],
                                       [param_dict['mean']]])), dim=1)

    def forward(self, in_feat):
        """Calculates harmonic contribution of bond/angle interactions to energy

        Parameters
        ----------
        in_feat: torch.Tensor
            input features, such as bond distances or angles of size (n,k), for
            n examples and k features.

        Returns
        -------
        energy: torch.Tensor
            output energy of size (n,1) for n examples.

        """

        n = len(in_feat)
        energy = torch.sum(self.harmonic_parameters[0, :] *
                           (in_feat - self.harmonic_parameters[1, :]) ** 2,
                           1).reshape(n, 1) / 2
        return energy


def LinearLayer(
        d_in,
        d_out,
        bias=True,
        activation=None,
        dropout=0,
        weight_init='xavier',
        weight_init_args=None,
        weight_init_kwargs=None):
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
