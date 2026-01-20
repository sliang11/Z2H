import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader


def get_dataloader(X, y):
    X_tensor, y_tensor = torch.from_numpy(X), torch.from_numpy(y)
    return DataLoader(TensorDataset(X_tensor, y_tensor), shuffle=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False