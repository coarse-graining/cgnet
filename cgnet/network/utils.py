# Authors: Nick Charron, Brooke Husic


import torch
import torch.nn as nn


def lipschitz_projection(model):
    """Performs L2 Lipschitz Projection via spectral normalization

    Parameters
    ----------
    model : Net() instance
        model to perform Lipschitz projection upon

    """
    lipschitz = 4.0
    for layer in model.arch:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            u, s, v = torch.svd(weight)
            if next(model.parameters()).is_cuda:
                lip_reg = torch.max(((s[0]) / lipschitz),
                                    torch.tensor([1.0]).cuda())
            else:
                lip_reg = torch.max(((s[0]) / lipschitz),
                                    torch.tensor([1.0]))
            layer.weight.data = weight / (lip_reg)


def dataset_loss(model, loader):
    """Compute average loss over arbitrary loader/dataset

    Parameters
    ----------
    model : Net() instance
        model to calculate loss
    loader : torch.utils.data.DataLoader() instance
        loader (with associated dataset)

    Returns
    -------
    loss : torch.Variable
        loss computed over the dataset

    """
    loss = 0
    num_batch = 0
    for num, batch in enumerate(loader):
        coords, force = batch
        U, F = model.forward(coords)
        loss += model.criterion(F, force)
        num_batch += 1
    loss /= num_batch
    return loss.data.item()
