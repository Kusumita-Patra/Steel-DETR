from torch.utils.tensorboard import SummaryWriter


def create_writer(log_dir):

    return SummaryWriter(log_dir)