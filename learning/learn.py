
import torch

from tqdm import tqdm
from dfw.losses import set_smoothing_enabled
import utils
from learn_utils import *

def train(loss, optimizer, device, dataset, model, lossfn, disable_bn, model_init, train_loader, \
    scheduler=None, epoch=0, weight_decay=0.0):
    model.train()
    optimizer = set_optimizer(optimizer)
    criterion = set_loss(loss)

    if disable_bn:
        batchnorm_mode(model, train=False)

    mult=0.5 if lossfn=='mse' else 1
    metrics = AverageMeter()


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if args.lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)

        if 'mnist' in dataset:
            data=data.view(data.shape[0],-1)

        output = model(data)
        loss = mult*criterion(output, target) + regularization(model, weight_decay, l2=True)

        metrics.update(n=data.size(0), loss=loss.item(), accuracy=accuracy(output, target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log_metrics(mode, metrics, epoch)
    logger.append('train' if mode=='train' else 'test', epoch=epoch, loss=metrics.avg['loss'], error=metrics.avg['error'],
                  lr=optimizer.param_groups[0]['lr'])
    print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    return metrics

def test()