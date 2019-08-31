# Authors: Nick Charron

import torch.nn as nn
from cgnet.feature import (GeometryFeature, Geometry, SchnetFeature,
                           RadialBasisFunction)
import warnings
g = Geometry(method='torch')


class FeaturePipeline(nn.Module):
    """ Class for combining sequential features requiring
    inter-layer transforms

    Attributes
    ----------
	layer_list : nn.ModuleList
		List of layers through which input data will
		pass.
	transforms : list of type.FunctionType or type.LambdaType
		List of declared functions or anonymous (lambda)
		functions that define transforms for data before each
		corresponding layer in the layer_list

    Notes
    -----
    Unlike FeatureCombiner(), this class does not retain
    intermediate output or offer residual connections.

    """

    def __init__(self, layer_list, transform_list):
        """ Initialization

        Parameters
        ----------
        layer_list : list of nn.Module instances
            List of layers through which input data will
            pass.
        transform_list : list of type.FunctionType or type.LambdaType
            List of declared functions or anonymous (lambda)
            functions that define transforms for data before each
            corresponding layer in the layer_list
        """

        super(FeaturePipeline, self).__init__()
        self.layer_list = nn.ModuleList(*layer_list)
        self.transforms = transforms

    def forward(self, in_features):
        """ Zip-forward through features and transforms

        Parameters
        ----------
        in_features : torch.Tensor
            Input features to the beginning of the layer list. Size
            can vary depending on layer type and the nature of
            transforms.

        Return
        ------
        out_features : torch.Tensor
            Output features at the end of the layer list. Size can
            vary depending on layer type and the nature of the
            transforms.
        """

        out_features = in_features
        for layer, transform in zip(self.layer_list, self.transforms):
            if transform != None:
                for sub_transform in transform:
                    out_features = sub_transform(out_features)
            out_features = layer(out_features)
        return out_features

class FeatureCombiner(nn.Module):
    """Class for combining GeometryFeatures and SchnetFeatures

    Attributes
    ----------
    layer_list : nn.ModuleList
        feature layers with which data is transformed before being passed to
        densely/fully connected layers prior to sum pooling and energy
        prediction/force generation.
    transforms : list of None or method types
        inter-featurelayer transforms that may be needed during the forward
        method. For example, SchnetFeature tools require a redundant form for
        distances, so outputs from a previous GeometryFeature layer must be
        re-indexed.
    save_geometry  : boolean
        specifies whether or not to save the output of GeometryFeature
        layers. It is important to set this to true if CGnet priors
        are to be used, and need to callback to GeometryFeature outputs.
    mappings : dictionary of strings
        dictionary of mappings to provide for specified inter-feature
        transforms. Keys are strings which describe the mapping, and values
        are mapping objects. For example, a redundant distance mapping may be
        represented as:

            {'redundant_distance_maping' : self.redundant_distance_mapping}
    """

    def __init__(self, layer_list, save_geometry=True, distance_indices=None):
        """Initialization

        Parameters
        ----------
        layer_list : list of nn.Module objects
            feature layers with which data is transformed before being passed to
            densely/fully connected layers prior to sum pooling and energy
            prediction/force generation.
        save_geometry : boolean (default=True)
            specifies whether or not to save the output of GeometryFeature
            layers. It is important to set this to true if CGnet priors
            are to be used, and need to callback to GeometryFeature outputs.
        distance_indices : list or np.ndarray of int (default=None)
            Indices of distances output from a GeometryFeature layer, used
            to isolate distances for redundant re-indexing for Schnet utilities
        """

        super(FeatureCombiner, self).__init__()
        self.layer_list= nn.ModuleList(layer_list)
        if type(save_geometry) == bool:
            self.save_geometry = save_geometry
        else:
            raise ValueError("save_geometry must be a boolean value")
        self.transforms = []
        self.mappings = {}
        self.distance_indices = distance_indices
        for layer in self.layer_list:
            if isinstance(layer, SchnetFeature):
                if (layer.calculate_geometry and any(isinstance(layer,
                    GeometryFeature) for layer in self.layer_list)):
                    warnings.warn(("This SchnetFeature has been set to "
                    "calculate pairwise distances. Set "
                    "SchnetFeature.calculate_geometry=False if you are "
                    "preceding this SchnetFeature with a GeometryFeature "
                    "in order to prevent unnecessarily repeated pairwsie "
                    "distance calculations"))
                    self.transforms.append(None)
                elif layer.calculate_geometry:
                    self.transforms.append(None)
                else:
                    if self.distance_indices is None:
                        raise RuntimeError(("Distance indices must be "
                                            "supplied to FeatureCombiner "
                                            "for redundant re-indexing."))
                    self.mappings['redundant_distance_mapping'] = (
                        g.get_redundant_distance_mapping(layer._distance_pairs))
                    self.transforms.append([self.distance_reindex])
            else:
                self.transforms.append(None)


    def distance_reindex(self, geometry_output):
        """Reindexes GeometryFeature distance outputs to redundant form for
        SchnetFeatures and related tools.

        Parameters
        ----------
        geometry_ouput : torch.Tensor
            geometrical feature output frome a GeometryFeature layer, of shape
            [n_frames, n_features].

        Returns
        -------
        redundant_distances : torch.Tensor
            pairwise distances transformed to shape
            [n_frames, n_beads, n_beads-1].
        """
        distances = geometry_output[:, self.distance_indices]
        return distances[:, self.mappings['redundant_distance_mapping']]

    def forward(self, coords, embedding_property=None):
        """Forward method through specified feature layers. The forward
        operation proceeds through self.layer_list in that same order
        as the input layer_list for __init__().

        Parameters
        ----------
        coords : torch.Tensor
            Input cartesian coordinates of size [n_frames, n_beads, 3]
        embedding_property : torch.Tensor (default=None)
            Some property that should be embedded. Can be nuclear charge
            or maybe an arbitrary number assigned for amino-acids.
            Size [n_frames, n_properties].

        Returns
        -------
        feature_ouput : torch.Tensor
            output tensor, of shape [n_frames, n_features] after featurization
            through the layers contained in self.layer_list.
        geometry_features : torch.Tensor (default=None)
            if save_geometry is True and the layer list is not just a single
            GeometryFeature layer, the output of the last GeometryFeature
            layer is returned alongside the terminal features for prior energy
            callback access. Else, None is returned.
        """
        feature_output = coords
        geometry_features = None
        for num, (layer, transform) in enumerate(zip(self.layer_list,
                                                     self.transforms)):
            if transform != None:
                # apply transform(s) before the layer if specified
                for sub_transform in transform:
                    feature_output = sub_transform(feature_output)
            if isinstance(layer, SchnetFeature):
                feature_output = layer(feature_output, embedding_property)
            else:
                feature_output = layer(feature_output)
            if isinstance(layer, GeometryFeature) and self.save_geometry:
                geometry_features = feature_output
        return feature_output, geometry_features
