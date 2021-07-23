import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


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

def set_optimizer(args, parameters):
    """
    - SGD
    - Adam
    - Adagrad
    - AMSGrad
    """
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.l2,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.l2)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.eta, weight_decay=args.l2)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.l2, amsgrad=True)
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.gamma = 1
    optimizer.eta = args.eta

    if args.load_opt:
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))

    return optimizer

def set_loss(args):
    #if args.loss == 'svm':
    #    loss_fn = MultiClassHingeLoss()
    if args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    # add losses
    else:
        raise ValueError

    print('L2 regularization: \t {}'.format(args.l2))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
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
'''
def get_model(args):
    assert args.dataset in ('cifar10', 'cifar100')

    if args.densenet:
        model = DenseNet3(args.depth, args.n_classes, args.growth, bottleneck=bool(args.bottleneck))
    elif args.wrn:
        model = WideResNet(args.depth, args.n_classes, args.width)
    else:
        raise NotImplementedError

    if args.load_model:
        state = torch.load(args.load_model)['model']
        new_state = OrderedDict()
        for k in state:
            # naming convention for data parallel
            if 'module' in k:
                v = state[k]
                new_state[k.replace('module.', '')] = v
            else:
                new_state[k] = state[k]
        model.load_state_dict(new_state)
        print('Loaded model from {}'.format(args.load_model))

    # Number of model parameters
    args.nparams = sum([p.data.nelement() for p in model.parameters()])
    print('Number of model parameters: {}'.format(args.nparams))

    if args.cuda:
        if args.parallel_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    return model
'''