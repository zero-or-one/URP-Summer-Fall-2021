import numpy as np
import os
from PIL import Image
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from data_utils import *
import torch.utils.data as data
import torchvision.datasets as datasets
from random import uniform, gauss

root = os.path.expanduser('~/data')

seed = 1000

def create_loaders(dataset_train, dataset_val, dataset_test,
                   train_size, val_size, test_size, batch_size, test_batch_size,
                   cuda, num_workers, split=True):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}
    if split:
        train_indices, val_indices = random_part((train_size, val_size),
                                                    len(dataset_train),
                                                    seed=seed)
    else: # shuffle
        train_size = train_size if train_size is not None else len(dataset_train)
        train_indices, = random_part((train_size,),
                                        len(dataset_train),
                                        seed=seed)
        val_size = val_size if val_size is not None else len(dataset_val)
        val_indices, = random_part((val_size,),
                                      len(dataset_val),
                                      seed=seed)
    test_size = test_size if test_size is not None else len(dataset_test)
    test_indices, = random_part((test_size,),
                                   len(dataset_test),
                                   seed=seed)
    # get subset of data from torch dataset
    dataset_train = Subset(dataset_train, train_indices)
    dataset_val = Subset(dataset_val, val_indices)
    dataset_test = Subset(dataset_test, test_indices)
    print('Dataset sizes: \t train: {} \t val: {} \t test: {}'
          .format(len(dataset_train), len(dataset_val), len(dataset_test)))
    print('Batch size: \t {}'.format(batch_size))
    train = data.DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True, **kwargs)
    val = data.DataLoader(dataset_val,
                                 batch_size=test_batch_size,
                                 shuffle=False, **kwargs)
    test = data.DataLoader(dataset_test,
                                  batch_size=test_batch_size,
                                  shuffle=False, **kwargs)
    train.tag = 'train'
    val.tag = 'val'
    test.tag = 'test'
    return train, val, test

def get_dataset(dataset, batch_size=16, cuda=0,
                  train_size=50, val_size=10, test_size=10,
                  test_batch_size=10, **kwargs):

    if dataset == 'mnist':
        loader_train, loader_val, loader_test = mnist(dataset=dataset, batch_size=batch_size, cuda=cuda,
                  train_size=train_size, val_size=val_size, test_size=test_size,
                  test_batch_size=test_batch_size, **kwargs)
    elif dataset == 'cifar100':
        loader_train, loader_val, loader_test = cifar(dataset=dataset, batch_size=batch_size, cuda=cuda,
                  train_size=train_size, val_size=val_size, test_size=test_size,
                  test_batch_size=test_batch_size, **kwargs)
    elif dataset == 'cifar10':
        loader_train, loader_val, loader_test = cifar(dataset=dataset, batch_size=batch_size, cuda=cuda,
                  train_size=train_size, val_size=val_size, test_size=test_size,
                  test_batch_size=test_batch_size, **kwargs)
    elif dataset == 'fashion-mnist':
        loader_train, loader_val, loader_test = fashion_mnist(dataset=dataset, batch_size=batch_size, cuda=cuda,
                  train_size=train_size, val_size=val_size, test_size=test_size,
                  test_batch_size=test_batch_size, **kwargs)
    else:
        raise NotImplementedError

    return loader_train, loader_val, loader_test

# define each dataset
def mnist(dataset, batch_size=64, cuda=0,
                  train_size=500, val_size=100, test_size=100,
                  test_batch_size=100, **kwargs):
    assert dataset == 'mnist'
    dataset_train = datasets.MNIST(root=root, train=True, transform=mnist_transform(True), download=True)
    dataset_val = datasets.MNIST(root=root, train=True, transform=mnist_transform(False))
    dataset_test = datasets.MNIST(root=root, train=False, transform=mnist_transform(False))

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, num_workers=0)

def cifar(dataset, batch_size=64, cuda=0,
                  train_size=500, val_size=100, test_size=100,
                  test_batch_size=100, **kwargs):
    assert dataset in ('cifar10', 'cifar100')
    dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    dataset_train = dataset(root=root, train=True,
                            transform=cifar_transform(True), download=True)
    dataset_val = dataset(root=root, train=True,
                          transform=cifar_transform(False))
    dataset_test = dataset(root=root, train=False,
                           transform=cifar_transform(False))

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, num_workers=4)

def fashion_mnist(dataset, batch_size=64, cuda=0,
                  train_size=500, val_size=100, test_size=100,
                  test_batch_size=100, **kwargs):
    assert dataset == 'fashion-mnist'
    dataset = datasets.FashionMNIST
    dataset_train = dataset(root=root, train=True,
                            transform=fashionMnist_transform(True), download=True)
    dataset_val = dataset(root=root, train=True,
                          transform=fashionMnist_transform(False))
    dataset_test = dataset(root=root, train=False,
                           transform=fashionMnist_transform(False))
    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, num_workers=4)

# Data points

def generate_gauss(meanx, meany, stdx, stdy, N):
    x = []
    y = []
    for i in range(N):
        xi = gauss(meanx, stdx)
        yi = gauss(meany, stdy)
        x.append(xi)
        y.append(yi)

    return x,y

def clusters():
    x1, y1 = generate_gauss(1, 2, 1, 1, 1000)
    x2, y2 = generate_gauss(0, 0, 1, 1, 1000)
    return (x1, y1), (x2, y2)

def half_doughnuts(lim1, lim2):
    N_1 = 0
    N_2 = 0
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    while (N_1 < lim1):
        x = uniform(-13.1, 13.1)
        y = uniform(-0.1, 13.1)
        if (y * y + x * x <= 169) & (y * y + x * x >= 49):
            x_1.append(x)
            y_1.append(y)
            N_1 += 1
    while (N_2 < lim2):
        x = uniform(-3.1, 23.1)
        y = uniform(-9.1, 4.1)
        if ((y - 4) * (y - 4) + (x - 10) * (x - 10) <= 169) & ((y - 4) * (y - 4) + (x - 10) * (x - 10) >= 49):
            x_2.append(x)
            y_2.append(y)
            N_2 += 1
    return (x_1, y_1), (x_2, y_2)