import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import json
from collections import defaultdict
import os
import errno
import pickle
import random
import matplotlib.pyplot as plt

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

def set_loss(loss, cuda=None):
    """
    - Mean average error loss (mae)
    - Mean square error loss (mse)
    - Negative log-likelihood (nll)
    - Cross entropy loss (ceo)
    - Multiclass hinge embedding (svm)
    - Kullback-Leibler divergence (kld)
    """
    if loss == 'mae':
        loss_fn = nn.L1Loss()
    elif loss == 'mse':
        loss_fn = nn.MSELoss()
    elif loss == 'nll':
        loss_fn == nn.NLLLoss()
    elif loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif loss == 'svm':
        loss_fn = nn.MultiClassHingeLoss()
    elif loss == 'kld':
        loss_fn = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError

    print('\nLoss function:')
    print(loss_fn)

    if cuda is not None:
        loss_fn = loss_fn.cuda()

    return loss_fn

def set_optimizer(opt, parameters, eta, decay, momentum=None, load_opt=False):
    """
    - SGD (sgd)
    - ASGD (asgd)
    - RMSPROP (rmsprop)
    - Adam (adam)
    - Adagrad (amsgrad)
    - Adamw (adamw)
    """
    if opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=eta, weight_decay=decay,
                                    momentum=momentum, nesterov=bool(momentum))
    elif opt == 'asgd':
        optimizer = torch.optim.ASGD(parameters, lr=eta, weight_decay=decay,
                                    momentum=momentum, nesterov=bool(momentum))
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=eta, alpha=0.99,
                                        eps=1e-08, weight_decay=decay, momentum=momentum, centered=False)
    elif opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=eta, weight_decay=decay)
    elif opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=eta, weight_decay=decay)
    elif opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=eta, weight_decay=decay, amsgrad=True)
    elif opt == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=eta, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay, amsgrad=False)
    else:
        raise ValueError(opt)

    print("Optimizer: \t ", opt)

    optimizer.gamma = 1
    # optimizer can be loaded
    if load_opt:
        state = torch.load(load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(load_opt))

    return optimizer

def set_scheduler(sch, optimizer, step_size, gamma=0.1, last_epoch=-1):
    if sch:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)
    else:
        scheduler = None
    return scheduler

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
                'optimizer': optimizer.state_dict()}, os.path.join(filename))

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()

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

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        #print("validation loss ", val_loss)
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def plot_curves(logger, valid):
    epochs = []
    loss = []
    err = []
    loss_val = []
    err_val = []
    for dict in logger.get('train'):
        epochs.append(dict['epoch'])
        loss.append(dict['loss'])
        err.append(dict['error'])
    if valid:
        for dict in logger.get('test'):
            loss_val.append(dict['loss'])
            err_val.append(dict['error'])
    plt.figure(figsize=(10, 5))
    plt.title("Loss Curve")
    plt.plot(loss_val, label="val")
    plt.plot(loss, label="train")
    plt.xlabel("epoches")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.title("Error Curve")
    plt.plot(err_val, label="val")
    plt.plot(err, label="train")
    plt.xlabel("epoches")
    plt.ylabel("Error")
    plt.legend()
    plt.show()