import os
import sys

sys.path.append("../URP")
path = ".."
sys.path.append(path)
sys.path.append(path + "/learning")
sys.path.append(path + "/data")
sys.path.append(path + "/models")


from data.data_utils import *
#from learning.learn import *

#from data_utils import *
#from learn import *
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

class FD(object):
    ''' Feature Destruction '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.name = "FD"
        self.noise = AddNoise(mean=self.mean, std=self.std)

    def forget_image(self, model, imglab, ds, loss, optimizer, epochs, device, dataset, lossfn,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
        '''Forget single image'''
        set_seed()
        if imglab is None:
            imglab = get_random_img(ds)
        img, lab = imglab
        plab = predict(model, img)
        print('-' * 20)
        print("True label is: ", lab)
        print("Predicted label is: ", plab)
        if plab != lab:
            print('-' * 20)
            print("PREDICTION IS WRONG, CHOOSE OTHER IMAGE")
            return
            #sys.exit()
        print('-' * 20)
        print("INITIAL D PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=ds, at_epoch=None)
        noisy_img = self.noise.encodes(img)
        data = ForgetDataset(noisy_img, lab)
        dataloader = DataLoader(data, batch_size=1)
        print('-' * 20)
        print("FORGETTING PROCESS")
        fine_tune(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=dataloader, val_loader=ds,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print("FINAL D PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=ds, at_epoch=None)
        plab = predict(model, img)
        print('-' * 20)
        print("True label is: ", lab)
        print("Predicted label is: ", plab)

    def forget_subset(self, model, img_num, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
        '''Forget random subset of data, either from single class or not.
           May be not working with this method'''
        set_seed()
        if class_id:
            forget_train, retain_train = remove_class(train_loader, [class_id])
            #forget_val, retain_val = remove_class(val_loader, [class_id])
            subset, left = separate_random(forget_train, img_num)
            forget = self.noise.encode_data(subset)
            retain = combine_datasets(left, retain_train)
            train = combine_datasets(forget, retain)
        else:
            subset, retain = separate_random(train_loader, img_num)
            forget = self.noise.encode_data(subset)
            train = combine_datasets(forget, retain)
        print('-' * 20)
        print("INITIAL Df PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget, at_epoch=None)
        print('-' * 20)
        print("INITIAL Dr PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain, at_epoch=None)
        print('-' * 20)
        print("FORGETTING PROCESS")
        fine_tune(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=train, val_loader=retain,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain, at_epoch=None)

    def forget_class(self, model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
        '''Forget whole class identity'''
        set_seed()
        forget_train, retain_train = remove_class(train_loader, [class_id])
        forget_val, retain_val = remove_class(val_loader, [class_id])
        noisy_forget = self.noise.encode_data(forget_train)
        concat = combine_datasets(noisy_forget, retain_train)
        print('-' * 20)
        print("INITIAL Df PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print("INITIAL Dr PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)
        print('-' * 20)
        print("FORGETTING PROCESS")
        fine_tune(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=concat, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)



class NIA(object):
    ''' Negative Information Allocation '''
    def __init__(self):
        self.mean = 0
        self.std = 0

    def forget_class(self, model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="NIA"):

        forget_train, retain_train = remove_class(train_loader, [class_id])
        forget_val, retain_val = remove_class(val_loader, [class_id])

        print('-' * 20)
        print("INITIAL Df PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print("INITIAL Dr PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)
        print('-' * 20)
        print("TRAINING ENCODER")
        #!!!!!!!!
        encoder_model = Encoder()
        train_encoder(encoder_model, model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=forget_train, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print("FORGETTING PROCESS")
        #encoder_model = model_learning(encoder_model, True)
        noisy_forget = encoder_model(forget_train)
        concat = combine_datasets(noisy_forget, retain_train)
        fine_tune(encoder_model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=concat, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)


class BL(object):
    '''Backward learning'''
    def __init__(self):
        pass


def train_encoder(encoder_model, model, loss, optimizer, epochs, device, dataset, lossfn, train_forget, val_forget,  train_retain, val_retain,
                  scheduler=None, weight_decay=0.0, lr=0.001, momentum=0.9, curves=True, patience=7,
                  min_delta=-1, step_size=10, gamma=0.5, name=""):
    encoder_model.to(device)
    model.to(device)
    model = model_learning(model, False)
    optimizer = set_optimizer(optimizer, encoder_model.parameters(), lr, weight_decay, momentum)
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
        model, _ = encoder_epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, encoder_model=encoder_model,
                                  lossfn=lossfn, forget_loader=train_forget, retain_loader=train_retain, scheduler=scheduler,
                                  weight_decay=0.0, epoch_num=ep, train=True, logger=logger)
        if (val_forget is not None):
            model, _ = encoder_epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset,
                                      model=model, encoder_model=encoder_model,
                                      lossfn=lossfn, forget_loader=val_forget, retain_loader=val_retain,
                                      scheduler=scheduler,
                                      weight_decay=0.0, epoch_num=ep, train=False, logger=logger)
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
    save_state(encoder_model, optimizer, filename)
    print("FINISHED TRAINING")
    if curves:
        plot_curves(logger, bool(val_forget))
    model = model_learning(model, True)

def encoder_epoch(criterion, optimizer, device, dataset, model, encoder_model, lossfn, forget_loader, retain_loader, logger, scheduler=None, weight_decay=0.0, epoch_num=10, train=True, otype="other"):
    if train:
        encoder_model.train()
    else:
        encoder_model.eval()

    mult=0.5 if lossfn=='mse' else 1
    metrics = AverageMeter()

    train_loader = (forget_loader, retain_loader)
    for batch_idx, ((data_forget, target_forget), (data_retain, target_retain)) in enumerate(train_loader):
        data_forget, target_forget = data_forget.to(device), target_forget.to(device)
        data_retain, target_retain = data_retain.to(device), target_retain.to(device)

        target = target_forget + target_retain

        if 'mnist' in dataset:
            data_forget = data_forget.view(data_forget.shape[0],-1)
            data_retain = data_retain.view(data_retain.shape[0], -1)

        if train:
            encoded = encoder_model(data_forget)
            both = encoded + data_retain
            output = model(both)
            if otype == "float":
                output = output.float()
                target = target.float()
            loss = mult * criterion(output, target) + regularization(encoder_model, weight_decay, l2=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            encoded = encoder_model(data_forget)
            both = encoded + data_retain
            output = model(both)
            if otype == "float":
                loss = mult * criterion(output.float(), target.float()) + regularization(encoder_model, weight_decay, l2=True)
            else:
                loss = mult * criterion(output, target) + regularization(encoder_model, weight_decay, l2=True)

    metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
    if epoch_num is not None:
        log_metrics('train' if train else 'test', metrics, epoch_num)
        logger.append('train' if train else 'test', epoch=epoch_num, loss=metrics.avg['loss'], error=metrics.avg['error'],
                      lr=optimizer.param_groups[0]['lr'])
    else:
        print("Loss: ", loss.item())
        print("Error: ", get_error(output, target))
    #print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    return encoder_model, metrics


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
    filename = f"./learning/checkpoints/{model.__class__.__name__}_{name}{ep+1}.pth.tar"
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
    filename = f"./learning/checkpoints/{model.__class__.__name__}_{name}{ep+1}.pth.tar"
    save_state(model, optimizer, filename)
    print("FINISHED TRAINING")
    print("Forget time is:", time.time() - start_time)

