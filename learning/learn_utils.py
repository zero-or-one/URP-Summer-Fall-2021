import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import json


def class_accuracy(pred_labels, true_labels):
    return sum(y_pred == y_act for y_pred, y_act in zip(pred_labels, true_labels))

def regularization(model, lamb=0.01, l2=True):
    if l2:
        regular = 0.5 * lamb * sum([p.data.norm() ** 2 for p in model.parameters()])
    else:
        regulr = 0.5 * lamb * sum([np.abs(p.data.norm()) for p in model.parameters()])
    return regular

def configure_learning_rate(optimizer, decay=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay
        # update state
        optimizer.eta = optimizer.param_groups[0]['lr']
    else:
        raise ValueError

def set_optimizer(opt, parameters, eta, decay, load_opt, momentum=None):
    """
    - SGD
    - Adam
    - Adagrad
    """
    if opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=eta, weight_decay=decay,
                                    momentum=momentum, nesterov=bool(momentum))
    elif opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=eta, weight_decay=decay)
    elif opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=eta, weight_decay=decay)
    elif opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=eta, weight_decay=decay, amsgrad=True)
    else:
        raise ValueError(opt)

    print("Optimizer: \t ", opt)

    optimizer.gamma = 1
    optimizer.eta = eta

    if load_opt:
        state = torch.load(load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(load_opt))

    return optimizer

def set_scheduler(sch, optimizer, step_size, gamma=0.1, last_epoch=-1):
    if sch:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    else:
        scheduler = None
    return scheduler

def set_loss(loss, cuda=None):
    #if args.loss == 'svm':
    #    loss_fn = MultiClassHingeLoss()
    if loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    # add losses
    else:
        raise ValueError

    print('\nLoss function:')
    print(loss_fn)

    if cuda is not None:
        loss_fn = loss_fn.cuda()

    return loss_fn


def batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    # optional in my code
    for l in model.children():
        batchnorm_mode(l, train=train)

def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

def log_metrics(split, metrics, epoch, **kwargs):
    print(f'[{epoch}] {split} metrics:' + json.dumps(metrics.avg))
