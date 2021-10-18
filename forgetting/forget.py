import os
import sys

sys.path.append("../URP")
path = ".."
sys.path.append(path)
sys.path.append(path + "/learning")
sys.path.append(path + "/data")
sys.path.append(path + "/models")

from learning.learn_utils import *
from learning.learn import *
import time
import torch
#from data.data_utils import AddNoise, remove_random, remove_class, combine_datasets
#from learning.learn import *
from data_utils import AddNoise, remove_random, remove_class, combine_datasets
from learn import *
from forget_utils import *

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

###########################
#    Proposed methods     #
###########################

def FD(class_id, mean, std, model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="FD"):
    ''' Feature Destruction '''
    set_seed()
    forget_train, retain_train = remove_class(train_loader, [class_id])
    forget_val, retain_val = remove_class(val_loader, [class_id])
    noise = AddNoise(mean=mean, std=std)
    noisy_forget = noise.encode_data(forget_train)
    #noisy_retain = noise.encode_data(retain)
    concat = combine_datasets(noisy_forget, retain_train)
    print('-' * 20)
    print("INITIAL Df PERFOMANCE")
    _ = test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
             test_loader=forget_val, at_epoch=None)
    print('-' * 20)
    print("INITIAL Dr PERFOMANCE")
    _ = test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
             test_loader=retain_val, at_epoch=None)
    print('-' * 20)
    print("FORGETTING PROCESS")
    _ = fine_tune(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                  train_loader=concat, val_loader=forget_val,
                  scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name='fine_tune')
    print('-' * 20)
    print('FINAL Df PERFOMANCE')
    _ = test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
             test_loader=forget_val, at_epoch=None)
    print('-' * 20)
    print('FINAL Dr PERFOMANCE')
    _ = test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
             test_loader=retain_val, at_epoch=None)


def NIA(class_id, mean, std, model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="NIA"):
    ''' Negative Information Allocation '''
    set_seed()
    pass
    # do we need to shuffle dataset???

def forget_image():
    set_seed()
    pass



#########################
#    Simple methods     #
#########################

def fine_tune(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="fine_tune"): # and retrain
    start_time = time.time()
    train(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler, weight_decay, lr, momentum, name)
    print("Forget time is:", time.time() - start_time)

def random_labels(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader, scheduler=None,
    weight_decay=0.0, lr=0.01, momentum=0.9, name="random_lables"):
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
            loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)
            loss.backward()
            optimizer.step()

        metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))

        log_metrics('train', metrics, ep)
        logger.append('train',epoch=ep, loss=metrics.avg['loss'],
                      error=metrics.avg['error'],
                      lr=optimizer.param_groups[0]['lr'])
        if (val_loader is not None):
            epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
                  train_loader=val_loader, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=False,
                  logger=logger)
        print(f'Epoch number: {ep} :\n Epoch Time: {np.round(time.time()-t,2)} sec')
    filename = f"./checkpoints/{model.__class__.__name__}_{name}{ep+1}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")
    print("Forget time is:", time.time() - start_time)

def hiding(model_base, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader, 
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="hiding"):
    start_time = time.time()
    #for p in model_base.parameters():
    #    if p.requires_grad:
    #        print(p.name, p.data)
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()
        model_base.cuda()
        out = model_base(img)
        #print(out.size())
        out_size = out.size()
        break
    modules=list(model_base.children())[:-1]
    removed = list(model_base.children())[-1]
    last_layer = torch.nn.Sequential(*modules)
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()
        last_layer.cuda()
        out = last_layer(img)
        in_size = out.size()
        break  
    #print(in_size[1])
    model = torch.nn.Sequential(*modules, torch.nn.Linear(int(in_size[1]), int(out_size[1]-1)))
    train(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler, weight_decay, lr, momentum, name)
    print("Forget time is:", time.time() - start_time)
    return model

def neg_gradient(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader, 
    scheduler=None, weight_decay=0.0, lr=0.001, momentum=0.9, name="neg_gradient"):
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
        if (val_loader is not None):
            epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
                  train_loader=val_loader, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=False,
                  logger=logger)
        print(f'Epoch number: {ep} :\n Epoch Time: {np.round(time.time()-t,2)} sec')
    filename = f"./checkpoints/{model.__class__.__name__}_{name}{ep+1}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")
    print("Forget time is:", time.time() - start_time)

