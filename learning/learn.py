
import torch

import utils
from learn_utils import *
import os
import sys
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epoch(criterion, optimizer, device, dataset, model, lossfn, disable_bn, train_loader, scheduler=None, weight_decay=0.0, epoch_num, train=True):
    if train:
        model.train()
    else:
        model.eval()

    if disable_bn:
        batchnorm_mode(model, train=False)

    mult=0.5 if lossfn=='mse' else 1

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if 'mnist' in dataset:
            data=data.view(data.shape[0],-1)

        output = model(data)
        loss = mult*criterion(output, target) + regularization(model, weight_decay, l2=True)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #log_metrics(mode, metrics, epoch)
    #logger.append('train' if mode=='train' else 'test', epoch=epoch_num, loss=metrics.avg['loss'], error=metrics.avg['error'],
    #              lr=optimizer.param_groups[0]['lr'])
    print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    #return metrics

def train(name, loss, optimizer, scheduler, epochs, device, dataset, model, lossfn, disable_bn, train_loader,
    weight_decay=0.0):

    optimizer = set_optimizer(optimizer)
    criterion = set_loss(loss)
    scheduler = set_scheduler(scheduler, optimizer, step_size=3, gamma=0.1, last_epoch=-1)

    for ep in range(epochs):

        t = time.time()
        epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
              disable_bn=disable_bn, train_loader=train_loader, scheduler=scheduler, weight_decay=0.0, epoch=ep, train=True)

        if epoch % 5 == 0:
            epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
              disable_bn=disable_bn, train_loader=train_loader, scheduler=scheduler, weight_decay=0.0, epoch=ep, train=True)
        #if epoch % 1 == 0:
        print(f'Epoch Time: {np.round(time.time()-t,2)} sec')
    torch.save(model.state_dict(), f"checkpoints/{name}_{ep}.pt")

def test(criterion, device, dataset, model, lossfn, disable_bn, train_loader, scheduler=None, weight_decay=0.0 train=True):

    #model.load_state_dict(torch.load(PATH))
    print("TESTING")
    epoch(criterion=criterion, optimizer=None, device=device, dataset=dataset, model=model, lossfn=lossfn,
              disable_bn=disable_bn, train_loader=train_loader, scheduler=scheduler, weight_decay=0.0, epoch=ep, train=True)