# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm, Jiang Wang

import torch
import torch.nn as nn


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
        \"Distances\", \"Angles\", \"Dihedral_cosines\", and/or
        \"Dihedral_sines\"
    feature_type: str
        features type from which to select coordinates.

    Examples
    --------
    To assemble the feat_dict input for a HarmonicLayer prior for bonds from an
    instance of a stats = ProteinBackboneStatistics():

    features = stats.get_bond_constants(flip_dict=True, zscores=True)
    bonds = dict((k, features[k]) for k in [(i, i+1) for i in
                  range(stats.n_beads)])
    bond_layer = HarmonicLayer(bonds, stats.descriptions, "Distances")

    """

    def __init__(self, feat_data, descriptions, feature_type):
        super(_PriorLayer, self).__init__()
        if feature_type not in descriptions.keys():
            raise ValueError('Feature type not found in descriptions')
        self.params = []
        self.feature_type = feature_type
        self.features = [feat for feat in feat_data.keys()]
        self.feat_idx = []
        # get number of each feature to determine starting idx
        nums = [len(descriptions['Distances']),
                len(descriptions['Angles']),
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

        raise NotImplementedError('forward() method must be overridden in \
                                custom classes inheriting from _PriorLayer()')


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
        for param_dict in self.params:
            if (key in param_dict for key in ('ex_vol', 'exp')):
                pass
            else:
                raise KeyError(
                    'Missing or incorrect key for repulsion parameters'
                )
        self.repulsion_parameters = torch.tensor([])
        for param_dict in self.params:
            self.repulsion_parameters = torch.cat((
                self.repulsion_parameters,
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
        dictionary defining the CG beads and interaction parameters for
        computing the energy contributions of the residual prior energy. The
        keys are tuples defining the CG beads involved in each pairwise
        interaction, and the values are dictionaries of physical constants
        involved in the corresponding harmonic interaction: The keys of this
        subdictionary are \"k\", and \"mean\", which are the harmonic spring
        constant (in energy/length^2 for bonds or energy units for angles) and
        the mean (in length units for bonds or dimensionless for angles)
        respectively. The corresponding values are the the numerical values of
        each constant. For example, for one such feat_dict entry:

            { (3, 4) : {  \"k\" : 139.2, \"mean\" : 1.2 }}

    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys. Possible
        feature keys are those implemented in ProteinBackBoneStatistics():
        \"Distacnces\", \"Angles\", \"Dihedral_cosines\", and/or
        \"Dihedral_sines\"
    feature_type: str
        features type from which to select coordinates.

    Notes
    -----
    This prior energy is useful for constraining the CGnet potential in regions
    of configuration space in which sampling is normally precluded by physical
    harmonic constraints assocaited with the structural integrity of the protein
    along its backbone. The harmonic parameters are also easily estimated from
    all atom simluation data because bond and angle distributions typically have
    Gaussian structure, which is easily intepretable as a harmonic energy
    contribution via the Boltzmann distribution.

    """

    def __init__(self, feat_data, descriptions=None, feature_type=None):
        super(HarmonicLayer, self).__init__(feat_data,
                                            descriptions=descriptions,
                                            feature_type=feature_type)
        for param_dict in self.params:
            if (key in param_dict for key in ('k', 'mean')):
                pass
            else:
                KeyError('Missing or incorrect key for harmonic parameters')
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


class ZscoreLayer(nn.Module):
    """Layer for Zscore normalization

    Parameters
    ----------
    zscores: torch.Tensor
        [2, n_features] tensor, where the first row contains the means
        and the second row contains the standard deviations of each
        feature

    Notes
    -----
    Zscore normalization can accelerate training convergence if placed
    after a ProteinBackboneFeature() layer, especially if the input features
    span different orders of magnitudes, such as the combination of angles
    and distances.

    For more information, see the documentation for
    sklearn.preprocessing.StandardScaler

    """

    def __init__(self, zscores):
        super(ZscoreLayer, self).__init__()
        self.zscores = zscores

    def forward(self, in_feat):
        """Normalizes each feature by subtracting its mean and dividing by
           its standard deviation.

        Parameters
        ----------
        in_feat: torch.Tensor
            input data of shape [n_frames, n_features]

        Returns
        -------
        rescaled_feat: torch.Tensor
            Zscore normalized features. Shape [n_frames, n_features]

        """
        rescaled_feat = (in_feat - self.zscores[0, :])/self.zscores[1, :]
        return rescaled_feat
