import torch
import numpy as np
import random


def set_seed(seed):
    if seed is None:
        np.random.seed(None)
        seed = np.random.randint(1e5)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


