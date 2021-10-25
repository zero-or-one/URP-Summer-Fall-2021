import torch
import numpy as np
import random
from collections import defaultdict
import json
import os
import errno
import pickle



def set_seed(seed=13):
    if seed is None:
        np.random.seed(None)
        seed = np.random.randint(1e5)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("SEED SET TO: ", seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

def log_metrics(split, metrics, epoch, **kwargs):
    print(f'[{epoch}] {split} metrics:' + json.dumps(metrics.avg))

def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    '''Make a log during the training and save it
    '''
    def __init__(self, index=None, path='logs/', always_save=True):
        if index is None:
            index = '{:06x}'.format(random.getrandbits(6 * 4))
        self.index = index
        self.filename = os.path.join(path, '{}.p'.format(self.index))
        self._dict = {}
        self.logs = []
        self.always_save = always_save

    def __getitem__(self, k):
        return self._dict[k]

    def __setitem__(self,k,v):
        self._dict[k] = v

    @staticmethod
    def load(filename, path='logs/'):
        if not os.path.isfile(filename):
            filename = os.path.join(path, '{}.p'.format(filename))
        if not os.path.isfile(filename):
            raise ValueError("{} is not a valid filename".format(filename))
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self):
        with open(self.filename,'wb') as f:
            pickle.dump(self, f)

    def get(self, _type):
        l = [x for x in self.logs if x['_type'] == _type]
        l = [x['_data'] if '_data' in x else x for x in l]
        # if len(l) == 1:
        #     return l[0]
        return l

    def append(self, _type, *args, **kwargs):
        kwargs['_type'] = _type
        if len(args)==1:
            kwargs['_data'] = args[0]
        elif len(args) > 1:
            kwargs['_data'] = args
        self.logs.append(kwargs)
        if self.always_save:
            self.save()
