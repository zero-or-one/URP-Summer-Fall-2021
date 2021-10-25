
from learn_utils import *
import os
import sys
import inspect
import time
sys.path.append("../URP")
from utils import *


def epoch(criterion, optimizer, device, dataset, model, lossfn, train_loader, logger, scheduler=None, weight_decay=0.0, epoch_num=10, train=True, otype="other"):
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
            output = model(data)
            if otype == "float":
                output = output.float()
                target = target.float()
            loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            output = model(data)
            if otype == "float":
                loss = mult * criterion(output.float(), target.float()) + regularization(model, weight_decay, l2=True)
            else:
                loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)

    metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
    if epoch_num is not None:
        log_metrics('train' if train else 'test', metrics, epoch_num)
        logger.append('train' if train else 'test', epoch=epoch_num, loss=metrics.avg['loss'], error=metrics.avg['error'],
                      lr=optimizer.param_groups[0]['lr'])
    else:
        print("Loss: ", loss.item())
        print("Error: ", get_error(output, target))
    #print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    return model, metrics


def train(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader, scheduler=None,
    weight_decay=0.0, lr=0.001, momentum=0.9, curves=True, patience=7, min_delta=-1, step_size=10, gamma=0.5, name=""):
    model.to(device)
    optimizer = set_optimizer(optimizer, model.parameters(), lr, weight_decay, momentum)
    criterion = set_loss(loss)
    scheduler = set_scheduler(scheduler, optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)

    mkdir('logs')
    mkdir('checkpoints')

    early_stop_callback = EarlyStopping(patience=patience, min_delta=min_delta)
    logger = Logger(index=str(model.__class__.__name__)+'_training')
    #logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index+'.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index+'_{}.pth')
    print("[Logging in {}]".format(logger.index))

    for ep in range(epochs):
        t = time.time()
        #configure_learning_rate(optimizer, epoch)
        model, _ = epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
               train_loader=train_loader, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=True, logger=logger)
        if (val_loader is not None):
            epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
                  train_loader=val_loader, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=False,
                  logger=logger)
            loss_val = logger.get('test')[-1]["loss"]
            #print(loss_val)
            early_stop_callback(loss_val)
        if scheduler:
            scheduler.step()
        print(f'Epoch number: {ep} \nEpoch Time: {np.round(time.time()-t,2)} sec')
        if early_stop_callback.early_stop:
            print("EARLY STOPPING")
            break
    filename = f"./learning/checkpoints/{model.__class__.__name__}_{name}{ep+1}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")
    if curves:
        plot_curves(logger, bool(val_loader))
    #return model

def test(model, loss, optimizer, device, dataset, lossfn, test_loader, at_epoch, name=""):
    model.to(device)
    optimizer = set_optimizer(optimizer, model.parameters(), 0.001, 0.001, 0.8)
    
    if at_epoch is not None:
        checkpoint = torch.load(f"checkpoints/{model.__class__.__name__}_{name}{at_epoch}.pth.tar")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = set_loss(loss)
    logger = Logger(index=str(model.__class__.__name__) + '_testing')
    print("TESTING")
    model, _ = epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, lossfn=lossfn,
            train_loader=test_loader, scheduler=None, weight_decay=0.0, epoch_num=at_epoch, train=False, logger=logger)
    print("FINISHED TESTING")
    #return model

def predict(model, img):
  img.unsqueeze_(0)
  return torch.argmax(model(img))