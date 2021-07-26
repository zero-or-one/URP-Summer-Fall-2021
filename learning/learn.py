
import torch

from tqdm import tqdm
from dfw.losses import set_smoothing_enabled
import utils
import learn_utils

def train(args, model, model_init, train_loader, scheduler=None, epoch=0, weight_decay=0.0):
    model.train()
    optimizer = set_optimizer(args)
    criterion = set_loss(args)

    if args.disable_bn:
        batchnorm_mode(model, train=False)

    mult=0.5 if args.lossfn=='mse' else 1
    metrics = AverageMeter()


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)

        if args.lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)

        if 'mnist' in args.dataset:
            data=data.view(data.shape[0],-1)

        output = model(data)
        loss = mult*criterion(output, target) + regularization(model, weight_decay, l2=True)

        if args.l1:
            l1_loss = sum([p.norm(1) for p in model.parameters()])
            loss += args.weight_decay * l1_loss

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