import torch


def save_checkpoint(state, filename):

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:

        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint