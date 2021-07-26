import torch
import numpy as np
import random
import os
import json
import errno


def set_seed(seed):
    if seed is None:
        np.random.seed(None)
        seed = np.random.randint(1e5)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

def log_metrics(split, metrics, epoch, **kwargs):
    print(f'[{epoch}] {split} metrics:' + json.dumps(metrics.avg))
