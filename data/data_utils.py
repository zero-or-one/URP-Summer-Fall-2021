from typing import List, Union

import os
import copy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
seed = 42
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

# show several random images
def show_random(dataset, num):
    for test_images, test_labels in dataset:
        n_samples = len(test_labels)
        for _ in range(num):
            rand_idx = int(np.random.random() * n_samples)
            sample_image = test_images[rand_idx]
            sample_label = test_labels[rand_idx]
            print(sample_image.size())
            img = sample_image.view(32, 32, -1)
            print("Label: ", sample_label)
            plt.axis('off')
            plt.imshow(img)
            plt.show()
        break

# Transforms
def general_transforms(train=True):
    #img_size = (64, 64) if resnet else (32, 32)
    transform_train = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform = transform_train if train else transform_test
    return transform

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

def cifar_transform(train=True):
    transform_train = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform = transform_train if train else transform_test
    return transform

def fashionMnist_transform(train=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor()
    ])
    transform = transform_train if train else transform_test
    return transform

class AddNoise():
    def __init__(self, mean=0., std=1., **kwargs):
        self.std = std
        self.mean = mean
        super().__init__(**kwargs)

    def encodes(self, x):
        return x + torch.randn(x.size()) * self.std + self.mean

    def generate(self, size=[1, 32, 32]):
        return torch.randn(size) * self.std + self.mean

    def encode_data(self, ds):
        imgs = []
        labs = []
        for img, label in ds:
            for l, di in enumerate(img):
                di = self.encodes(di)
                imgs.append(di)
                labs.append(label[l])
        dataset = ForgetDataset(imgs, labs)
        dataloader = DataLoader(dataset, batch_size=ds.batch_size)
        return dataloader

class ForgetDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

    def __remove__(self, remove_list):
        data = np.delete(self.data, remove_list, axis=0)
        targets = np.delete(self.targets, remove_list, axis=0)
        return data, targets

def remove_random(ds, num): # dummy, but no time for good one
    '''Remove random images from dataset'''
    count = 0
    imgs = []
    labs = []
    batch_size = ds.batch_size
    for (img, lab) in ds:
        for di in img:
            imgs.append(di)
            count+=1
        for dl in lab:
            labs.append(dl)
        if count >= num-1:
            break
    dataset = ForgetDataset(imgs, labs)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def remove_class(ds, class_id): # dump again
    '''Remove class or several classes from dataset'''
    forget_imgs = []
    forget_labs = []
    retain_imgs = []
    retain_labs = []
    for (img, lab) in ds:
        for idx, dl in enumerate(lab):
            if dl in class_id:
                forget_labs.append(dl)
                forget_imgs.append(img[idx])
            else:
                retain_labs.append(dl)
                retain_imgs.append(img[idx])
    forget = ForgetDataset(forget_imgs, forget_labs)
    retain = ForgetDataset(retain_imgs, retain_labs)
    forget = DataLoader(forget, batch_size=ds.batch_size)
    retain = DataLoader(retain, batch_size=ds.batch_size)
    return forget, retain

def combine_datasets(ds1, ds2, shuffle=False, device='cuda'): # dummy iterable approach
    imgs = []
    labs = []
    for (img, lab) in ds1:
        img, lab = img.to(device), lab.to(device)
        for di in img:
            imgs.append(di)
        for dl in lab:
            labs.append(dl)
    for (img, lab) in ds2:
        img, lab = img.to(device), lab.to(device)
        for di in img:
            imgs.append(di)
        for dl in lab:
            labs.append(dl)    
    ds = ForgetDataset(imgs, labs)
    ds = DataLoader(ds, batch_size=ds1.batch_size, shuffle=shuffle)
    return ds

def get_random_img(ds):
    n_batch = len(ds)
    rand_batch = int(np.random.random() * n_batch)
    #imgs, labs = ds[rand_batch]
    for i, (s_imgs, s_labs) in enumerate(ds):
        if i < rand_batch:
            continue
        imgs, labs = s_imgs, s_labs
        break
    n_imgs = len(imgs)
    rand_idx = int(np.random.random() * n_imgs)
    img = imgs[rand_idx]
    lab = labs[rand_idx]
    return img, lab

def get_random(ds, num):
    imgs = []
    labs = []
    for i in range(num):
        img, lab = get_random_img(ds)
        imgs.append(img)
        labs.append(lab)
    dataset = ForgetDataset(imgs, labs)
    dataloader = DataLoader(dataset, ds.batch_size)
    return dataloader

# Sorry, I don't have time to use efficient algorithms, let's go with straightforward instead
def separate_random(ds, num):
    got_imgs = []
    got_labs = []
    left_imgs = []
    left_labs = []
    max_id = len(ds) * ds.batch_size
    rand_idxs = np.random.randint(0, max_id, num)
    idx = 0
    for imgs, labs in ds: # dummy
        for i, img in enumerate(imgs):
            lab = labs[i]
            if idx in rand_idxs:
                got_imgs.append(img)
                got_labs.append(lab)
            else:
                left_imgs.append(img)
                left_labs.append(lab)
            idx+=1
    got = ForgetDataset(got_imgs, got_labs)
    left = ForgetDataset(left_imgs, left_labs)
    got = DataLoader(got, ds.batch_size)
    left = DataLoader(left, ds.batch_size)
    return got, left

'''
def separate_data(ds, given=False, idxs=[], target=0, cuda=0):
    labels_all = []
    idxs_all = []
    for imgs, labels in ds:
        if not given:
            idxs = np.concatenate((idxs, dummy), axis=None)
            dummy = (labels == target).nonzero().numpy()
            idxs = np.concatenate((idxs, dummy), axis=None)
            print(idxs)
        labels_all = np.concatenate((labels_all, labels), axis=None)
        print(labels_all)
    batch_size = ds.batch_size
    kwargs = {'num_workers': ds.num_workers, 'pin_memory': True} if cuda else {}
    forget = Subset(ds, idxs)
    idxs_ret = np.copy(idxs)
    for idx in idxs:
        idxs_ret = idxs_ret[idxs_ret != id]
    #idxs_ret = np.setdiff1d(labels_all, idxs)
    retain = Subset(ds, idxs_ret)
    forget = data.DataLoader(forget, batch_size=batch_size,
                                   shuffle=True, **kwargs)
    retain = data.DataLoader(retain, batch_size=batch_size,
                                   shuffle=True, **kwargs)
    return forget, retain
'''