import numpy as np
import os
from PIL import Image
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from data_utils import *
import torch.utils.data as data
import torchvision.datasets as datasets
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
        train_indices, = random_part(train_size,
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

# define each dataset
def mnist(dataset, batch_size=28, cuda=0,
                  train_size=50, val_size=10, test_size=10,
                  test_batch_size=10, **kwargs):

    assert dataset == 'mnist'

    dataset_train = datasets.MNIST(root=root, train=True, transform=mnist_transform(), download=True)
    dataset_val = datasets.MNIST(root=root, train=True, transform=mnist_transform(False))
    dataset_test = datasets.MNIST(root=root, train=False, transform=mnist_transform(False))

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, num_workers=0)

def cifar(dataset, batch_size, cuda,
                  train_size=45000, augment=True, val_size=5000, test_size=10000,
                  test_batch_size=128, **kwargs):

    assert dataset in ('cifar10', 'cifar100')


    dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    dataset_train = dataset(root=root, train=True,
                            transform=cifar_transform(True))
    dataset_val = dataset(root=root, train=True,
                          transform=cifar_transform(False))
    dataset_test = dataset(root=root, train=False,
                           transform=cifar_transform(False))

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, num_workers=4)


def imagenet(dataset, batch_size, cuda,
                 train_size=63257, augment=False, val_size=10000, test_size=26032,
                 test_batch_size=1000, **kwargs):

    assert dataset == 'imagenet'

    dataset = datasets.SVHN
    dataset_train = dataset(root=root, split='train',
                            transform=imagenet_transforms(True))
    dataset_val = dataset(root=root, split='train',
                          transform=imagenet_transforms(False))
    dataset_test = dataset(root=root, split='test',
                           transform=imagenet_transforms(False))

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, num_workers=4)

'''
# get dataset from args option
def get_dataset(args):

    print('Dataset: \t {}'.format(args.dataset.upper()))

    # remove values if None
    for k in ('train_size', 'val_size', 'test_size'):
        if args.__dict__[k] is None:
            args.__dict__.pop(k)

    if args.dataset == 'mnist':
        loader_train, loader_val, loader_test = mnist(**vars(args))
    elif 'cifar' in args.dataset:
        loader_train, loader_val, loader_test = cifar(**vars(args))
    elif args.dataset == 'svhn':
        loader_train, loader_val, loader_test = svhn(**vars(args))
    else:
        raise NotImplementedError

    args.train_size = len(loader_train.dataset)
    args.val_size = len(loader_val.dataset)
    args.test_size = len(loader_test.dataset)

    return loader_train, loader_val, loader_test
'''
def get_dataset(dataset):

    if dataset == 'mnist':
        loader_train, loader_val, loader_test = mnist(dataset=dataset)
    elif dataset == 'cifar':
        loader_train, loader_val, loader_test = cifar(dataset=dataset)
    elif dataset == 'cifar10':
        loader_train, loader_val, loader_test = cifar10(dataset=dataset)
    elif dataset == 'imagenet':
        loader_train, loader_val, loader_test = imagenet(dataset=dataset)
    else:
        raise NotImplementedError

    return loader_train, loader_val, loader_test