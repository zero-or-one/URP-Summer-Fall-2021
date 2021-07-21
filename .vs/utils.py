import torch
import numpy as np
import torch.utils.data as data

# get subset by indices
class getPart(data.Dataset):
    def __init__(self, dataset, idxs=None):

        super(getPart, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

        if self.idxs is not None:
            self.size = len(self.idxs)
            assert self.size >= 0 and \
                self.size <= len(self.dataset), \
                "number of indices are out of range"
        else:
            self.n_samples = len(self.dataset)

    def get_size(self):
        return self.n_samples

    def get_item(self, idx):
        if torch.is_tensor(idx) and idx.dim():
            res = [self[i] for i in idx]
            return torch.stack([x[0] for x in res]), torch.LongTensor([x[1] for x in res])
        if self.idxs is None:
            return self.dataset[idx]
        else:
            return self.dataset[self.indices[idx]]

 # get random indices of certain size with unique option
def random_part(sizes, num, seed=None, unique=True):
    # save current random state
    state = np.random.get_state()
    sum_sizes = sum(size)
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

#########################################################
# Code reference:
#  https://www.programcreek.com/python/example/105103/torchvision.datasets.MNIST
#########################################################