import os
import argparse


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--disable-bn', action='store_true', default=False,
                        help='Put batchnorm in eval mode and don\'t update the running averages')
    parser.add_argument('--epochs', type=int, default=31, metavar='N',
                        help='number of epochs to train (default: 31)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--forget-class', type=int, default=None,
                        help='Class to forget')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='uses L1 regularizer instead of L2')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--num-to-forget', type=int, default=None,
                        help='Number of samples of class to forget')
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step-size', default=None, type=int, help='learning rate scheduler')
    parser.add_argument('--unfreeze-start', default=None, type=str,
                        help='All layers are freezed except the final layers starting from unfreeze-start')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    args = parser.parse_args()

    manual_seed(args.seed)

    if args.step_size == None: args.step_size = args.epochs + 1

    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_{str(args.filters).replace('.', '_')}_forget_{args.forget_class}"
        if args.num_to_forget is not None:
            args.name += f"_num_{args.num_to_forget}"
        if args.unfreeze_start is not None:
            args.name += f"_unfreeze_from_{args.unfreeze_start.replace('.', '_')}"
        if args.augment:
            args.name += f"_augment"
        args.name += f"_lr_{str(args.lr).replace('.', '_')}"
        args.name += f"_bs_{str(args.batch_size)}"
        args.name += f"_ls_{args.lossfn}"
        args.name += f"_wd_{str(args.weight_decay).replace('.', '_')}"
        args.name += f"_seed_{str(args.seed)}"
    print(f'Checkpoint name: {args.name}')

    mkdir('logs')

    logger = Logger(index=args.name + '_training')
    logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index + '.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index + '_{}.pth')

    print("[Logging in {}]".format(logger.index))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs('checkpoints', exist_ok=True)

    train_loader, valid_loader, test_loader = datasets.get_loaders(args.dataset, class_to_replace=args.forget_class,
                                                                   num_indexes_to_replace=args.num_to_forget,
                                                                   batch_size=args.batch_size, seed=args.seed,
                                                                   augment=args.augment)

    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    print(f"Number of Classes: {num_classes}")
    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)

    if args.model == 'allcnn':
        classifier_name = 'classifier.'
    elif 'resnet' in args.model:
        classifier_name = 'linear.'

    if args.resume is not None:
        state = torch.load(args.resume)
        state = {k: v for k, v in state.items() if not k.startswith(classifier_name)}
        incompatible_keys = model.load_state_dict(state, strict=False)
        assert all([k.startswith(classifier_name) for k in incompatible_keys.missing_keys])
    model_init = copy.deepcopy(model)
    torch.save(model.state_dict(), f"checkpoints/{args.name}_init.pt")

    parameters = model.parameters()
    if args.unfreeze_start is not None:
        parameters = []
        layer_index = 1e8
        for i, (n, p) in enumerate(model.named_parameters()):
            if (args.unfreeze_start in n) or (i > layer_index):
                layer_index = i
                parameters.append(p)

    weight_decay = args.weight_decay if not args.l1 else 0.
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss().to(args.device) if args.lossfn == 'ce' else torch.nn.MSELoss().to(
        args.device)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        t1 = time.time()
        run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay,
                  mode='train')
        #         run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='test')
        if epoch % 5 == 0:
            if not args.disable_bn:
                run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay,
                          mode='dry_run')
            run_epoch(args, model, model_init, test_loader, criterion, optimizer, scheduler, epoch, weight_decay,
                      mode='test')
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.name}_{epoch}.pt")
        print(f'Epoch Time: {np.round(time.time() - t1, 2)} sec')
