from typing import List, Union

import os
import copy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_data_loaders(args):

    print('Dataset: \t {}'.format(args.dataset.upper()))

    # remove values if None
    for k in ('train_size', 'val_size', 'test_size'):
        if args.__dict__[k] is None:
            args.__dict__.pop(k)

    if args.dataset == 'mnist':
        loader_train, loader_val, loader_test = loaders_mnist(**vars(args))
    elif 'cifar' in args.dataset:
        loader_train, loader_val, loader_test = loaders_cifar(**vars(args))
    elif args.dataset == 'svhn':
        loader_train, loader_val, loader_test = loaders_svhn(**vars(args))
    else:
        raise NotImplementedError

    args.train_size = len(loader_train.dataset)
    args.val_size = len(loader_val.dataset)
    args.test_size = len(loader_test.dataset)

    return loader_train, loader_val, loader_test