# Authors: Nick Charron, Brooke Husic

import torch
import torch.nn as nn
import torch.utils.data.DataLoader

def lipschitz_projection(model,strength=10.0):
    """Performs L2 Lipschitz Projection via spectral normalization

    Parameters
    ----------
    model : CGnet() instance
        model to perform Lipschitz projection upon
    strength : float (default=10.0)
        Strength of L2 lipschitz projection via spectral normalization.
        The magntitude of {dominant weight matrix eigenvalue / strength}
        is compared to unity, and the weight matrix is rescaled by the max
        of this comparison

    References
    ----------
    Gouk, H., Frank, E., Pfahringer, B., & Cree, M. (2018). Regularisation
    of Neural Networks by Enforcing Lipschitz Continuity. ArXiv:1804.04368
    [Cs, Stat]. Retrieved from http://arxiv.org/abs/1804.04368
    """
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            u, s, v = torch.svd(weight)
            if next(model.parameters()).is_cuda:
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]).cuda())
            else:
                lip_reg = torch.max(((s[0]) / strength),
                                    torch.tensor([1.0]))
            layer.weight.data = weight / (lip_reg)


def dataset_loss(model, loader):
    """Compute average loss over arbitrary loader/dataset

    Parameters
    ----------
    model : CGNet() instance
        model to calculate loss
    loader : torch.utils.data.DataLoader() instance
        loader (with associated dataset)

    Returns
    -------
    loss : torch.Variable
        loss computed over the dataset

    Example
    -------
    test_set = MoleculeDataset(coords[test_indices], forces[test_indices])
    test_sampler = torch.utils.data.RandomSubSetSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_sampler,
                                              batch_size=512)
    test_error = dataset_loss(MyModel, test_loader)

    """
    loss = 0
    num_batch = 0
    for num, batch in enumerate(loader):
        coords, force = batch
        potential, pred_force = model.forward(coords)
        loss += model.criterion(pred_force, force) * (coords.size()[0] /
                                                      loader.batch_size)
        num_batch += 1
    loss /= num_batch
    return loss.data.item()
