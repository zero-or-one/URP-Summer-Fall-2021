
import os
import sys
sys.path.append("../URP")
from learning.learn_utils import *
from learning.learn import *
from utils import *
import time
import torch
'''
Methods I need to implement

- Retraining
- Rapid retraining
- Fine-tuning
- Negative Gradiemnt
- Random Labels
- Hiding
- Fisher Forgetting
- Variational forgetting

'''

def fine_tune(model, loss, optimizer, epochs, device, dataset, lossfn, disable_bn, train_loader,
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9): # and retrain
    start_time = time.time()
    train(model, loss, optimizer, epochs, device, dataset, lossfn, disable_bn, train_loader, None,
    scheduler, weight_decay, lr, momentum)
    print("Forget time is:", time.time() - start_time)

def random_labels(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, scheduler=None,
    weight_decay=0.0, lr=0.01, momentum=0.9):
    start_time = time.time()
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
        configure_learning_rate(optimizer, epoch)
        t = time.time()
        model.train()

        mult = 0.5 if lossfn == 'mse' else 1
        metrics = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if 'mnist' in dataset:
                data = data.view(data.shape[0], -1)

            optimizer.zero_grad()
            output = model(data)
            loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)
            loss.backward()
            optimizer.step()

        metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))

        log_metrics('train', metrics, ep)
        logger.append('train',epoch=ep, loss=metrics.avg['loss'],
                      error=metrics.avg['error'],
                      lr=optimizer.param_groups[0]['lr'])

        print(f'Epoch number: {ep} :\n Epoch Time: {np.round(time.time()-t,2)} sec')
    filename = f"./checkpoints/{model.__class__.__name__}_{ep}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")
    print("Forget time is:", time.time() - start_time)

def hiding(model, loss, optimizer, epochs, device, dataset, lossfn, disable_bn, train_loader,
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
    start_time = time.time()
#    if "ResNet" in str(model.__class__.__name__):
#        pass
#    else:
    removed = model[-1]
    inp, out = removed.weight.size
    model = model[:-1]
    model = torch.nn.Sequential(*removed)
    model = torch.nn.Sequential(model, torch.nn.Linear(inp, out-1))
    train(model, loss, optimizer, epochs, device, dataset, lossfn, disable_bn, train_loader, None,
    scheduler, weight_decay, lr, momentum)
    print("Forget time is:", time.time() - start_time)

def neg_gradient(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader,
    scheduler=None, weight_decay=0.0, lr=0.001, momentum=0.9):
    start_time = time.time()
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
        #configure_learning_rate(optimizer, epoch)
        t = time.time()
        model.train()

        mult = 0.5 if lossfn == 'mse' else 1
        metrics = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if 'mnist' in dataset:
                data = data.view(data.shape[0], -1)

            optimizer.zero_grad()
            output = model(data)
            output = -1 * output
            loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)
            loss.backward()
            optimizer.step()

        metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))

        log_metrics('train', metrics, ep)
        logger.append('train',epoch=ep, loss=metrics.avg['loss'],
                      error=metrics.avg['error'],
                      lr=optimizer.param_groups[0]['lr'])

        print(f'Epoch number: {ep} :\n Epoch Time: {np.round(time.time()-t,2)} sec')
    filename = f"./checkpoints/{model.__class__.__name__}_{ep+1}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")
    print("Forget time is:", time.time() - start_time)

