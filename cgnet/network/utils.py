# Authors: Nick Charron, Brooke Husic, Jiang Wang
# Contributors: Dominik Lemm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import warnings

from cgnet.feature import GeometryFeature, SchnetFeature, FeatureCombiner


def _schnet_feature_linear_extractor(schnet_feature, return_weight_data_only=False):
    """Helper function to extract instances of nn.Linear from a SchnetFeature

    Parameters
    ----------
    schnet_feature : SchnetFeature instance
        The SchnetFeature instance from which nn.Linear instances will be
        extracted.
    return_weight_data_only : bool (default=False)
        If True, the function returns the torch tensor for each weight
        layer rather than the nn.Linear instance.

    Returns
    -------
    linear_list : list of nn.Linear instances or np.arrays,
        The list of nn.Linear layers extracted from the supplied
        SchnetFeature. See notes below for the order of nn.Linear instances
        in this list.
    weight_data : list of torch.Tensors
        If 'return_data=True', the function instead returns the torch tensors
        of each nn.Linear instance weight. See notes below for the order of
        tensors in this list

    Notes
    -----
    Each InteractionBlock contains nn.Linear instances in the following order:

    1. initial_dense layer
    2. cfconv.filter_generator layer 1
    3. cfconv.filter_generator layer 2
    4. output layer 1
    5. output layer 2

    This gives five linear layers in total per InteractionBlock. The order of
    the nn.Linear instances are returned by _schnet_feature_linear_extractor().
    This is a hardcoded choice, becasue we assume that architectural structure
    of all InteractionBlocks are exactly the same (i.e., 1-5 above).
    """

    linear_list = []
    for block in schnet_feature.interaction_blocks:
        for block_layer in [block.initial_dense,
                            block.cfconv.filter_generator,
                            block.output_dense]:
            linear_list += [layer for layer in block_layer
                            if isinstance(layer, nn.Linear)]
    if return_weight_data_only:
        weight_data = [layer.weight.data for layer in linear_list]
        return weight_data
    else:
        return linear_list


def lipschitz_projection(model, strength=10.0, network_mask=None, schnet_mask=None):
    """Performs L2 Lipschitz Projection via spectral normalization

    Parameters
    ----------
    model : cgnet.network.CGnet() instance
        model to perform Lipschitz projection upon
    strength : float (default=10.0)
        Strength of L2 lipschitz projection via spectral normalization.
        The magntitude of {dominant weight matrix eigenvalue / strength}
        is compared to unity, and the weight matrix is rescaled by the max
        of this comparison
    network_mask : None, 'all', or list of bool (default=None)
        mask used to exclude certain terminal network layers from lipschitz
        projection. If an element is False, the corresponding weight layer
        is exempt from a lipschitz projection. If set to all, a False mask
        is used for all terminal network weights. If None, all terminal network
        weight layers are subject to Lipschitz constraint.
    schnet_mask : None, 'all', or list of bool (default=None)
        mask used to exclude certain SchnetFeature layers from lipschitz projection.
        If an element is False, the corresponding weight layer is exempt from a
        lipschitz projection. The linear layers of a SchnetFeature InteractionBlock
        have the following arrangement:

        1. initial_dense layer
        2. cfconv.filter_generator layer 1
        3. cfconv.filter_generator layer 2
        4. output layer 1
        5. output layer 2

        that is, each InteractionBlock contains 5 nn.Linear instances. If set
        to 'all', a False mask is used for all weight layers in every
        InteractionBlock. If None, all weight layers are subject to Lipschitz
        constraint.

    Notes
    -----
    L2 Lipshitz regularization is a per-layer regularization that constrains
    the Lipschitz constant of each mapping from one linear layer to the next.
    As formulated by Gouk et. al. (2018), this constraint can be enforced by
    comparing the magnitudes between the weighted dominant singular value of
    the linear layer weight matrix and unity, taking the maximum, and
    normalizing the weight matrix by this result:

        W = W / max( s_dom / lambda, 1.0 )

    for weight matrix W, dominant singular value s_dom, and regularization
    strength lambda. In this form, a strong regularization is achieved for
    lambda -> 0, and a weak regularization is achieved for lambda -> inf.

    For nn.Linear weights that exist in SchnetFeatures (in the form of dense
    layers in InteractionBlocks and dense layers in the continuous filter
    convolutions), we assume that the architectural structure of all
    InteractionBlocks (and the continuous filter convolutions therein) is
    fixed to be the same - that is the nn.Linear instances always appear
    in SchnetFeatures in the following fixed order:

    1. initial_dense layer
    2. cfconv.filter_generator layer 1
    3. cfconv.filter_generator layer 2
    4. output layer 1
    5. output layer 2

    References
    ----------
    Gouk, H., Frank, E., Pfahringer, B., & Cree, M. (2018). Regularisation
    of Neural Networks by Enforcing Lipschitz Continuity. arXiv:1804.04368
    [Cs, Stat]. Retrieved from http://arxiv.org/abs/1804.04368
    """

    # Grab all instances of nn.Linear in the model, including those
    # that are part of SchnetFeatures
    # First, we grab the instances of nn.Linear from model.arch
    network_weight_layers = [layer for layer in model.arch
                             if isinstance(layer, nn.Linear)]
    # Next, we grab the nn.Linear instances from the SchnetFeature
    schnet_weight_layers = []
    # if it is part of a FeatureCombiner instance
    if isinstance(model.feature, FeatureCombiner):
        for feature in model.feature.layer_list:
            if isinstance(feature, SchnetFeature):
                schnet_weight_layers += _schnet_feature_linear_extractor(
                    feature)
    # Lastly, we handle the case of SchnetFeatures that are not part of
    # a FeatureCombiner instance
    elif isinstance(model.feature, SchnetFeature):
        schnet_weight_layers += _schnet_feature_linear_extractor(model.feature)

    # Next, we assemble a (possibly combined from terminal network and
    # SchnetFeature) mask
    if network_mask is None:
        network_mask = [True for _ in network_weight_layers]
    elif network_mask is 'all':
        network_mask = [False for _ in network_weight_layers]
    if network_mask is not None:
        if not isinstance(network_mask, list):
            raise ValueError("Lipschitz network mask must be list of booleans")
        if len(network_weight_layers) != len(network_mask):
            raise ValueError("Lipshitz network mask must have the same number "
                             "of elements as the number of nn.Linear "
                             "modules in the model.arch attribute.")

    if schnet_mask is None:
        schnet_mask = [True for _ in schnet_weight_layers]
    elif schnet_mask is 'all':
        schnet_mask = [False for _ in schnet_weight_layers]
    if schnet_mask is not None:
        if not isinstance(schnet_mask, list):
            raise ValueError("Lipschitz schnet mask must be list of booleans")
        if len(schnet_weight_layers) != len(schnet_mask):
            raise ValueError("Lipshitz schnet mask must have the same number "
                             "of elements as the number of nn.Linear "
                             "modules in the model SchnetFeature.")

    full_mask = network_mask + schnet_mask
    full_weight_layers = network_weight_layers + schnet_weight_layers
    for mask_element, layer in zip(full_mask, full_weight_layers):
        if mask_element:
            weight = layer.weight.data
            u, s, v = torch.svd(weight)
            if next(model.parameters()).is_cuda:
                device = weight.device
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]).to(device))
            else:
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]))
            layer.weight.data = weight / (lip_reg)


def dataset_loss(model, loader, optimizer=None,
                 regularization_function=None,
                 train_mode=True,
                 verbose_interval=None,
                 print_function=None):
    r"""Compute average loss over arbitrary data loader.
    This can be used during testing, in which `optimizer` and
    `regularization_function` will remain None, or it can be used
    during training, in which an optimizer and (optional)
    regularization_function are provided.

    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        model to calculate loss
    loader : torch.utils.data.DataLoader() instance
        loader (with associated dataset)
    optimizer : torch.optim method or None (default=None)
        If not None, the optimizer will be zeroed and stepped for each batch.
    regularization_function : in-place function or None (default=None)
        If not None, the regularization function will be applied after
        stepping the optimizer. It must take only "model" as its input
        and operate in-place.
    train_mode : bool (default=True)
        Specifies whether to put the model into train mode for training/learning
        or eval mode for testing/inference. See Notes about the important
        distinction between these two modes. The model will always be reverted
        back to training mode.
    verbose_interval : integer or None (default=None)
        If not None, a printout of the batch number and loss will be provided
        at the specified interval (with respect to batch number).
    print_function : python function or None (default=None)
        Print function that takes (batch_number, batch_loss) as its only
        two arguments, to print updates with our default or the style of
        your choice when verbose_interval is not None.

    Returns
    -------
    loss : float
        loss computed over the entire dataset. If the last batch consists of a
        smaller set of left over examples, its contribution to the loss is
        weighted by the ratio of number elements in the MSE matrix to that of
        the normal number of elements associated with the loader's batch size
        before summation to a scalar.

    Example
    -------
    from torch.utils.data import DataLoader

    # assume model is a CGNet object

    # For test data, no optimizer or regularization are needed
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    test_loss = dataset_loss(net, test_data_loader)

    # For training data, an optimizer is needed. Regularization may
    # be used, too
    training_data_loader = DataLoader(training_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Regularization must be in place
    def my_reg_fxn(model, strength=lipschitz_strength):
        lipschitz_projection(model, strength=strength)

    def my_print_fxn(batch_num, batch_loss):
        print("--> Batch #{}, loss = {}".format(batch_num, batch_loss))

    training_loss = dataset_loss(net, training_data_loader,
                                 optimizer = optimizer,
                                 regularization_function = my_reg_fxn,
                                 train_mode=True,
                                 verbose_interval = 128,
                                 print_function = my_print_fxn)

    Notes
    -----
    This method assumes that if there is a smaller batch, it will be at the
    end: namely, we assume that the size of the first batch is the largest
    batch size.

    It is important to use train_mode=False when performing inference/assessing
    a model on test data because certain PyTorch layer types, such as
    BatchNorm1d and Dropout, behave differently in 'eval' and 'train' modes.
    For more information, please see

        https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval

    """
    if optimizer is None:
        if regularization_function is not None:
            raise RuntimeError(
                "regularization_function is only used when there is an optimizer, "
                "but you have optimizer=None."
            )
        if train_mode:
            raise RuntimeError(
                "Without an optimizer, you probably wanted train_mode=False"
            )

    if train_mode:
        model.train()
    else:
        model.eval()

    loss = 0
    effective_number_of_batches = 0

    for batch_num, batch_data in enumerate(loader):
        if optimizer is not None:
            optimizer.zero_grad()

        coords, force, embedding_property = batch_data
        if batch_num == 0:
            reference_batch_size = coords.numel()

        batch_weight = coords.numel() / reference_batch_size
        if batch_weight > 1:
            raise ValueError(
                "The first batch was not the largest batch, so you cannot use "
                "dataset loss."
            )

        if loader.dataset.embeddings is not None:
            potential, predicted_force = model.forward(coords,
                                                       embedding_property=embedding_property)
        else:
            potential, predicted_force = model.forward(coords)

        batch_loss = model.criterion(predicted_force, force)

        if optimizer is not None:
            batch_loss.backward()
            optimizer.step()

            if regularization_function is not None:
                regularization_function(model)

        if verbose_interval is not None:
            if(batch_num + 1) % verbose_interval == 0:
                if print_function is None:
                    print("Batch: {}, Loss: {:.2f}".format(batch_num+1,
                                                           batch_loss))
                else:
                    print_function(batch_num+1, batch_loss)

        loss += batch_loss.cpu().detach().numpy() * batch_weight

        effective_number_of_batches += batch_weight

    loss /= effective_number_of_batches

    # If the model was in eval mode, put model back into training mode
    if model.training == False:
        model.train()

    return loss
