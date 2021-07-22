import numpy as np
import os
from PIL import Image
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from data_prep import *
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
    dataset_train = getPart(dataset_train, train_indices)
    dataset_val = getPart(dataset_val, val_indices)
    dataset_test = getPart(dataset_test, test_indices)

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


def mnist(dataset, batch_size=64, cuda=0,
                  train_size=5000, val_size=1000, test_size=1000,
                  test_batch_size=100, **kwargs):

    assert dataset == 'mnist'
    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    dataset_train = datasets.MNIST(root=root, train=True, transform=mnist_transform())
    dataset_val = datasets.MNIST(root=root, train=True, transform=mnist_transform(False))
    dataset_test = datasets.MNIST(root=root, train=False, transform=mnist_transform(False))

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, num_workers=0)

'''
def loaders_cifar(dataset, batch_size, cuda,
                  train_size=45000, augment=True, val_size=5000, test_size=10000,
                  test_batch_size=128, **kwargs):

    assert dataset in ('cifar10', 'cifar100')

    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if augment:
        print('Using data augmentation on CIFAR data set.')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        print('Not using data augmentation on CIFAR data set.')
        transform_train = transform_test

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    dataset_train = dataset(root=root, train=True,
                            transform=transform_train)
    dataset_val = dataset(root=root, train=True,
                          transform=transform_test)
    dataset_test = dataset(root=root, train=False,
                           transform=transform_test)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, num_workers=4)


def loaders_svhn(dataset, batch_size, cuda,
                 train_size=63257, augment=False, val_size=10000, test_size=26032,
                 test_batch_size=1000, **kwargs):

    assert dataset == 'svhn'

    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    # Data loading code
    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if augment:
        print('Using data augmentation on SVHN data set.')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        print('Not using data augmentation on SVHN data set.')
        transform_train = transform_test

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset = datasets.SVHN
    dataset_train = dataset(root=root, split='train',
                            transform=transform_train)
    dataset_val = dataset(root=root, split='train',
                          transform=transform_test)
    dataset_test = dataset(root=root, split='test',
                           transform=transform_test)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, num_workers=4)
'''

def get_data(args):

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