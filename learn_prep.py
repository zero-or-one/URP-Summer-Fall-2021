import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim

def class_accuracy(pred_labels, true_labels):
    return sum(y_pred == y_act for y_pred, y_act in zip(pred_labels, true_labels))

# L1 regularization
def l1_term(model, weight_decay):
    loss = sum([l.norm(1) for l in model.parameters()])
    return weight_decay * loss

# L2 regularization
def l2_term(model,model_init,weight_decay):
    loss = 0
    for (_, x), (_, x0) in zip(model.named_parameters(), model_init.named_parameters()):
        if x0.requires_grad:
            loss +=  ((x-x0) *  (x-x0)).sum()
    return loss * (weight_decay / 2.)

def regularization(model, l, l2=True):
    if l2:
        regular = 0.5 * l * sum([p.data.norm() ** 2 for p in model.parameters()])
    else:
        regulr = 0.5 * l * sum([np.abs(p.data.norm()) for p in model.parameters()])
    return regular

def configure_learning_rate(optimizer, epoch):
    pass

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