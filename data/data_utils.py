from typing import List, Union

import os
import copy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data


# get subset by indices
# it exists by default, redefined for comfort
class Subset(data.Dataset):
    def __init__(self, dataset, indices=None):

        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.num_samples = -1
        if self.indices is None:
            self.num_samples = len(self.dataset)
        else:
            self.num_samples = len(self.indices)

            assert self.num_samples >= 0 and \
                self.num_samples <= len(self.dataset), \
                "length of {} incompatible with dataset of size {}"\
                .format(self.num_samples, len(self.dataset))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx) and idx.dim():
            res = [self[i] for i in idx]
            return torch.stack([x[0] for x in res]), torch.LongTensor([x[1] for x in res])
        if self.indices is None:
            return self.dataset[idx]
        else:
            return self.dataset[self.indices[idx]]

 # get random indices of certain size with unique option
def random_part(sizes, num, seed=None, unique=True):
    # save current random state
    state = np.random.get_state()
    sum_sizes = sum(sizes)
    assert sum_sizes <= num, "subset size is out of range"

    np.random.seed(seed)
    subset = np.random.choice(num, size=sizes,
                                    replace=(not unique))
    perm = np.random.permutation(subset)
    res = []
    temp = 0
    for size in sizes:
        res.append(perm[temp: temp + size])
        temp += size
    # restore initial state
    np.random.set_state(state)
    return res

# Transforms

def mnist_transform(train=True):
    transform_train = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])
    transform = transform_train if train else transform_test
    return transform

