# Author: Nick Charron
# Contributors: Brooke Husic, Jiang Wang

import torch
import torch.nn as nn
import numpy as np

class PriorLayer(nn.Module):
    """Layer for adding prior energy computations external to CGnet hidden
    output
    Parameters
    ----------
    feat_data: list or dict
        list of tuples defining each feature from which to calculate
        interactions
    descriptions: dict
        dictionary of CG bead indices as tuples, for feature keys.
    feature_type: str
        features type from which to select coordinates.
    """

    def __init__(self, feat_data, excluded_volume=5.5, exponent=6.0,
                 descriptions=None, feature_type=None):
        super(PriorLayer, self).__init__()
        if descriptions and not feature_type:
            raise RuntimeError('Must declare feature_type if using \
                                descriptions')
        if descriptions and feature_type:
            self.feature_type = feature_type
            self.features = feat_data
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
            if isinstance(type(self.features),list):
                for feat in self.features:
                    self.feat_idx.append(self.start_idx +
                              descriptions[self.feature_type].index(feat))

    def forward(self, in_feat):
        raise NotImplementedError


class HarmonicLayer(PriorLayer):
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
        super(HarmonicLayer, self).__init__(feat_data, descriptions=None,
                                            feature_type=None)
            self.harmonic_parameters = torch.tensor([])
            for key, params in feat_data.items():
                self.features.append(key)
                self.feat_idx.append(start_idx +
                                     descriptions[self.feature_type].index(key))
                self.harmonic_parameters = torch.cat((self.harmonic_parameters,
                                           torch.tensor([[params['k']],
                                           [params['mean']]])), dim=1)

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
        energy = torch.sum(self.harmonic_parameters[0, :] * (in_feat -
                           self.harmonic_parameters[1, :]) ** 2,
                           1).reshape(n, 1) / 2
        return energy
