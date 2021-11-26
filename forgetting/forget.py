import os
import sys

sys.path.append("../URP")
path = ".."
sys.path.append(path)
sys.path.append(path + "/learning")
sys.path.append(path + "/data")
sys.path.append(path + "/models")


from data_utils import *
#from learning.learn import *

#from data_utils import *
#from learn import *
from forget_utils import *
from copy import deepcopy

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
        fine_tune_helper(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
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
        else:
            subset, retain = separate_random(train_loader, img_num)
            forget = self.noise.encode_data(subset)
        train = combine_datasets(forget, retain)
        print('-' * 20)
        print("INITIAL Df PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=subset, at_epoch=None)
        print('-' * 20)
        print("INITIAL Dr PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain, at_epoch=None)
        print('-' * 20)
        print("FORGETTING PROCESS")
        fine_tune_helper(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=train, val_loader=retain,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=subset, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain, at_epoch=None)

    def forget_image(self, model, imglab, ds, loss, optimizer, epochs, device, dataset, lossfn,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
        '''Forget single image'''
        set_seed()
        if imglab is None:
            imglab = get_random_img(ds)
        img, lab = imglab
        img0 = torch.clone(img)
        #img = img.to(device)
        #lab = lab.to(device)
        #model = model.to(device)
        plab = predict(model, img, device)
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
        data = ForgetDataset(noisy_img, [lab])
        dataloader = DataLoader(data, batch_size=1)
        print('-' * 20)
        print("FORGETTING PROCESS")
        fine_tune_helper(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=dataloader, val_loader=ds,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=self.name)
        print('-' * 20)
        print("FINAL D PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=ds, at_epoch=None)
        plab = predict(model, img0, device)
        print('-' * 20)
        print("True label is: ", lab)
        print("Predicted label is: ", plab)


class NIA(object):
    ''' Negative Information Allocation '''
    def __init__(self, input_size=1024, num_layer=2, hidden_size=1024):
        self.mean = 0
        self.std = 0
        self.encoder_model = Encoder(input_size=input_size, hidden_size=hidden_size, num_layer=num_layer)

    def forget_class(self, model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="NIA_class"):

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
        model = train_encoder(encoder_model=self.encoder_model, model=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_forget=forget_train, val_forget=forget_val, train_retain=retain_train, val_retain=retain_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name)
        print('-' * 20)
        print("FORGETTING PROCESS")
        #encoder_model = model_learning(encoder_model, True)
        noisy_data = self.encoder_model.predict(forget_train, device, dataset)
        #noisy_forget = ForgetDataset(noisy_data, target)
        #noisy_forget = DataLoader(dataset, retain_train.batch_size)
        concat = combine_datasets(noisy_data, retain_train, shuffle=True, device=device)
        fine_tune_helper(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=concat, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name, retain_graph=True)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)
        return model

    def forget_subset(self, model, img_num, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
        '''Forget random subset of data, either from single class or not.
           May be not working with this method'''
        set_seed()
        name="NIA_subset"
        if class_id:
            forget_train, retain_train = remove_class(train_loader, [class_id])
            #forget_val, retain_val = remove_class(val_loader, [class_id])
            subset, left = separate_random(forget_train, img_num)
            forget = subset
            retain = combine_datasets(left, retain_train)
        else:
            subset, retain = separate_random(train_loader, img_num)
            forget = subset
        #train = combine_datasets(forget, retain)
        print('-' * 20)
        print("INITIAL Df PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=subset, at_epoch=None)
        print('-' * 20)
        print("INITIAL Dr PERFOMANCE")
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain, at_epoch=None)
        print('-' * 20)
        print("TRAINING ENCODER")
        model = train_encoder(encoder_model=self.encoder_model, model=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_forget=forget, val_forget=None, train_retain=retain, val_retain=None,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name)
        print('-' * 20)
        print("FORGETTING PROCESS")
        #encoder_model = model_learning(encoder_model, True)
        noisy_data = self.encoder_model.predict(forget, device, dataset)
        #noisy_forget = ForgetDataset(noisy_data, target)
        #noisy_forget = DataLoader(dataset, retain_train.batch_size)
        concat = combine_datasets(noisy_data, retain, shuffle=True, device=device)
        fine_tune_helper(model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=concat, val_loader=subset,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name, retain_graph=True)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=subset, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain, at_epoch=None)
        return model

class BL(object):
    '''Backward learning'''
    def __init__(self):
        pass

    def forget_class(self, model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader,
                     val_loader,
                     scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9):
        '''Forget whole class identity'''
        set_seed()
        name = 'backward_learning'
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
        print("FORGETTING PROCESS")
        set_seed()
        start_time = time.time()
        model.to(device)
        optimizer = set_optimizer(optimizer, model.parameters(), lr, weight_decay, momentum)
        criterion = set_loss(loss)
        scheduler = set_scheduler(scheduler, optimizer, step_size=3, gamma=0.1, last_epoch=-1)

        mkdir('logs')
        mkdir('checkpoints')

        logger = Logger(index=str(model.__class__.__name__) + '_training')
        # logger['args'] = args
        logger['checkpoint'] = os.path.join('models/', logger.index + '.pth')
        logger['checkpoint_step'] = os.path.join('models/', logger.index + '_{}.pth')
        print("[Logging in {}]".format(logger.index))

        for ep in range(epochs//2): # double epoch
            # configure_learning_rate(optimizer, epoch)
            t = time.time()
            model.train()

            mult = 0.5 if lossfn == 'mse' else 1
            metrics = AverageMeter()

            for batch_idx, (data, target) in enumerate(forget_train):
                data, target = data.to(device), target.to(device)

                if 'mnist' in dataset:
                    data = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = model(data)
                loss = mult * criterion(output, target) + regularization(model, weight_decay, l2=True)
                # this one is changed
                ascent_loss = make_ascent(loss)
                ascent_loss.backward()
                optimizer.step()

            metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
            log_metrics('train', metrics, ep)
            logger.append('train', epoch=ep, loss=metrics.avg['loss'],
                          error=metrics.avg['error'],
                          lr=optimizer.param_groups[0]['lr'])
            if (forget_val is not None):
                epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model,
                      lossfn=lossfn,
                      train_loader=forget_val, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=False,
                      logger=logger)
            print(f'Epoch number: {2*ep} :\n Epoch Time: {np.round(time.time() - t, 2)} sec')
            t = time.time()
            epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model,
                             lossfn=lossfn,
                             train_loader=retain_train, scheduler=scheduler, weight_decay=0.0, epoch_num=2*ep+1, train=True,
                             logger=logger, retain_graph=False)
            if (val_loader is not None):
                epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model,
                      lossfn=lossfn,
                      train_loader=retain_val, scheduler=scheduler, weight_decay=0.0, epoch_num=ep, train=False,
                      logger=logger)
            print(f'Epoch number: {2 * ep + 1} :\n Epoch Time: {np.round(time.time() - t, 2)} sec')
        filename = f"./checkpoints/{model.__class__.__name__}_{name}{epochs + 1}.pth.tar"
        save_state(model, optimizer, filename)
        print("FINISHED TRAINING")
        print("Forget time is:", time.time() - start_time)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
             test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
             test_loader=retain_val, at_epoch=None)


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
        encoder_model, _ = encoder_epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset, model=model, encoder_model=encoder_model,
                                  lossfn=lossfn, forget_loader=train_forget, retain_loader=train_retain, scheduler=scheduler,
                                  weight_decay=0.0, epoch_num=ep, train=True, logger=logger)
        if (val_forget is not None):
            encoder_model, _ = encoder_epoch(criterion=criterion, optimizer=optimizer, device=device, dataset=dataset,
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
    filename = f"./checkpoints/{model.__class__.__name__}_{name}{ep+1}.pth.tar"
    save_state(encoder_model, optimizer, filename)
    print("FINISHED TRAINING")
    if curves:
        plot_curves(logger, bool(val_forget))
    model = model_learning(model, True)
    return model

def encoder_epoch(criterion, optimizer, device, dataset, model, encoder_model, lossfn, forget_loader, retain_loader, logger, scheduler=None, weight_decay=0.0, epoch_num=10, train=True, otype="other"):
    if train:
        encoder_model.train()
    else:
        encoder_model.eval()

    mult=0.5 if lossfn=='mse' else 1
    metrics = AverageMeter()

    for batch_idx, (forg, ret) in enumerate(zip(forget_loader, retain_loader)):
        (data_forget, target_forget) = forg
        (data_retain, target_retain) = ret
        data_forget, target_forget = data_forget.to(device), target_forget.to(device)
        data_retain, target_retain = data_retain.to(device), target_retain.to(device)

        target = torch.cat((target_forget, target_retain), 0)

        if 'mnist' in dataset:
            data_forget = data_forget.view(data_forget.shape[0],-1)
            data_retain = data_retain.view(data_retain.shape[0], -1)

        if train:
            encoded = encoder_model(data_forget, dataset)
            if dataset == 'mnist':
              encoded = encoded.reshape(data_forget.size(0),data_forget.size(1))
            else:
              encoded = encoded.reshape(data_forget.size(0),data_forget.size(1), data_forget.size(2), data_forget.size(3))
            #print(encoded.shape)
            #print(data_forget.shape)
            both = torch.cat((encoded, data_retain), 0)
            output = model(both)
            if otype == "float":
                output = output.float()
                target = target.float()
            loss = mult * criterion(output, target) + regularization(encoder_model, weight_decay, l2=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            encoded = encoder_model(data_forget, dataset)
            #encoded = encoded.reshape(data_forget.size(0),data_forget.size(1), data_forget.size(2), data_forget.size(3))
            both = torch.cat((encoded, data_retain), 0)
            #print(model)
            #print("in", both.shape)
            output = model(both)
            #print("out", output)
            #print("outsize", output.shape)
            #print("targ", target)
            #print("targsize", target.shape)
            if otype == "float":
                loss = mult * criterion(output.float(), target.float()) + regularization(encoder_model, weight_decay, l2=True)
            else:
                loss = mult * criterion(output, target) + regularization(encoder_model, weight_decay, l2=True)

    metrics.update(n=both.size(0), loss=loss.item(), error=get_error(output, target))
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

def retrain(model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="retrain"):
        
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
        print("RETRAINING")
        # reset the model
        model.apply(weight_reset)
        fine_tune_helper(model=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=retain_train, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name, patience=100)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)        
    
def fine_tune(model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="finetune"):
        
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
        print("FINETUNING")
        fine_tune_helper(model=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=retain_train, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)

def random_labels(model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="random_labels"):
        
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
        print("RANDOMING")
        random_labels_helper(model=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=retain_train, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)

def hiding(model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="random_labels"):
        
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
        print("HIDING")
        hiding_helper(model_base=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=retain_train, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)
                 
def neg_gradient(model, class_id, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
        scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="neg_grad", ver1=True):
        
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
        print("ASCENDING")
        if ver1 = True:
            neg_grad = neg_gradient_helper_ver1
        else:
            neg_grad = neg_gradient_helper_ver2
        neg_grad(model=model, loss=loss, optimizer=optimizer, epochs=epochs, device=device, dataset=dataset, lossfn=None,
                      train_loader=forget_train, val_loader=forget_val,
                      scheduler=scheduler, weight_decay=weight_decay, lr=lr, momentum=momentum, name=name)
        print('-' * 20)
        print('FINAL Df PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=forget_val, at_epoch=None)
        print('-' * 20)
        print('FINAL Dr PERFOMANCE')
        test(model=model, loss=loss, lossfn=lossfn, optimizer=optimizer, device=device, dataset=dataset,
                 test_loader=retain_val, at_epoch=None)

                 
# HELPERS
def fine_tune_helper(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="", retain_graph=False, patience=10): # and retrain
    start_time = time.time()
    train(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler, weight_decay, lr, momentum, name, retain_graph=retain_graph, patience=patience, min_delta=-1.5)
    print("Forget time is:", time.time() - start_time)
    
def random_labels_helper(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader, scheduler=None,
    weight_decay=0.0, lr=0.01, momentum=0.9, name="random_lables"):
    set_seed()
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

def hiding_helper(model_base, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader, 
    scheduler=None, weight_decay=0.0, lr=0.01, momentum=0.9, name="hiding"):
    set_seed()
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
    scheduler, weight_decay, lr, momentum, name, patience=10, min_delta=-1.5)
    print("Forget time is:", time.time() - start_time)
    return model

# few versions for negative gradient as it may be done in different ways
def neg_gradient_helper_ver1(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.001, momentum=0.9, name="neg_gradient", curves=True):
    set_seed()
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

def neg_gradient_helper_ver1(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.001, momentum=0.9, name="neg_gradient", curves=True):
    set_seed()
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
            # change output sign
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

def neg_gradient_helper_ver2(model, loss, optimizer, epochs, device, dataset, lossfn, train_loader, val_loader,
    scheduler=None, weight_decay=0.0, lr=0.001, momentum=0.9, name="neg_gradient", curves=True):
    set_seed()
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
            # this one is changed
            ascent_loss = make_ascent(loss)
            ascent_loss.backward()
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