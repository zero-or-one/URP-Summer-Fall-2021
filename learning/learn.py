
from learn_utils import *
import os
import sys
import inspect
import time
sys.path.append("../URP")
from utils import *


def epoch(criterion, optimizer, device, dataset, model, lossfn, train_loader, logger, scheduler=None, weight_decay=0.0, epoch_num=10, train=True):
    if train:
        model.train()
    else:
        model.eval()

    mult=0.5 if lossfn=='mse' else 1
    metrics = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if 'mnist' in dataset:
            data=data.view(data.shape[0],-1)

        if train:
            optimizer.zero_grad()
            output = model(data)
            loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)
            loss.backward()
            optimizer.step()
        else:
            output = model(data)
            loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)

    metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))

    log_metrics('train' if train else 'test', metrics, epoch_num)
    logger.append('train' if train else 'test', epoch=epoch_num, loss=metrics.avg['loss'], error=metrics.avg['error'],
                  lr=optimizer.param_groups[0]['lr'])
    #print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))

    return metrics


def train(model, loss, optimizer, scheduler, epochs, device, dataset, lossfn, train_loader, val_loader,
    weight_decay=0.0, lr=0.001, momentum=0.9):
    model.to(device)
    optimizer = set_optimizer(optimizer, model.parameters(), lr, weight_decay, momentum)
    criterion = set_loss(loss)
    scheduler = set_scheduler(scheduler, optimizer, step_size=3, gamma=0.1, last_epoch=-1)

    mkdir('logs')
    mkdir('checkpoints')

    logger = Logger(index=str(model.__class__.__name__)+'_training')
    #logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index+'.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index+'_{}.pth')
    print("[Logging in {}]".format(logger.index))

    for ep in range(epochs):
        t = time.time()
        epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
               train_loader=train_loader, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=True, logger=logger)
        if (ep % 5 == 0 and (val_loader is not None)):
            epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
                  train_loader=val_loader, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=False,
                  logger=logger)
        print(f'Epoch number: {ep} :\n Epoch Time: {np.round(time.time()-t,2)} sec')
    filename = f"./checkpoints/{model.__class__.__name__}_{ep+1}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")


def test(model, loss, optimizer, device, dataset, lossfn, test_loader, at_epoch):
    
    model.to(device)
    optimizer = set_optimizer(optimizer, model.parameters(), 0.001, 0.001, 0.8)
    checkpoint = torch.load(f"checkpoints/{model.__class__.__name__}_{at_epoch}.pth.tar")
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = set_loss(loss)
    logger = Logger(index=str(model.__class__.__name__) + '_testing')
    print("TESTING")
    epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
            train_loader=test_loader, scheduler=None, weight_decay=0.0, epoch_num=0, train=False, logger=logger)
    print("FINISHED TESTING")
