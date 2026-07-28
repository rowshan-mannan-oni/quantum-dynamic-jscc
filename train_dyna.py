# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.train_options import TrainOptions
import os
import torch
import torchvision
import torchvision.transforms as transforms

# Extract the options
opt = TrainOptions().parse()

# Prepare the dataset
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=opt.num_workers, drop_last=True)
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)


# Create the checkpoint folder
opt.name = 'C' + str(opt.C_channel) + '_L2_' + str(opt.lambda_L2) + '_re_' + str(opt.lambda_reward) + '_' + opt.select
path = os.path.join(opt.checkpoints_dir, opt.name)
if not os.path.exists(path):
    os.makedirs(path)

# Initialize the model
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

total_iters = 0                # the total number of training iterations
total_epoch = opt.n_epochs_joint + opt.n_epochs_decay + opt.n_epochs_fine


def apply_stage(epoch):
    """Put the model into the training stage that <epoch> belongs to.

    Derived from the epoch rather than triggered on a boundary, so it is
    correct no matter which epoch a run starts at. Switching on
    'epoch == boundary + 1' silently leaves a resumed run on the wrong
    learning rate, because that test is never true when resuming past it.
    """
    if epoch <= opt.n_epochs_joint:
        stage, lr, freeze = 'Joint learning', opt.lr_joint, False
    elif epoch <= opt.n_epochs_joint + opt.n_epochs_decay:
        stage, lr, freeze = 'Decay', opt.lr_decay, False
    else:
        stage, lr, freeze = 'Fine-tuning', opt.lr_fine, True

    model.optimizer_G.param_groups[0]['lr'] = lr
    for param in model.netP.parameters():
        param.requires_grad = not freeze
    for param in model.netSE.parameters():
        param.requires_grad = not freeze
    return stage, lr, freeze


def save_all(tag, completed_epoch):
    """Save the networks plus the state needed to resume from them."""
    model.save_networks(tag)
    model.save_training_state(tag, {'epoch': completed_epoch, 'temp': model.temp})


# Pick up where a previous run stopped
start_epoch = opt.epoch_count
if opt.continue_train:
    state = model.load_training_state(opt.epoch)
    if state is None:
        print('WARNING: no training state alongside the checkpoint. Weights were '
              'loaded, but the optimizer and the temperature restart from scratch.')
    else:
        start_epoch = state['epoch'] + 1
        model.temp = state['temp']
        print('Resuming after epoch %d: starting at epoch %d with temperature %.5f'
              % (state['epoch'], start_epoch, model.temp))

if start_epoch > total_epoch:
    print('Nothing to do: epoch %d is already past the %d epoch schedule.'
          % (start_epoch, total_epoch))

current_stage = None

for epoch in range(start_epoch, total_epoch + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

    # Select the learning rate and freezing for this epoch
    stage, lr, freeze = apply_stage(epoch)
    if stage != current_stage:
        print(f'{stage} stage begins!')
        print(f'Learning rate changed to {lr}')
        if freeze:
            print('Policy Network Disabled!')
        current_stage = stage

    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_input(data[0])         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
            for k, v in losses.items():
                message += '%s: %.5f ' % (k, v)
            print(message)  # print the message
            log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            # This epoch is unfinished, so resume by repeating it from the start
            save_all(save_suffix, epoch - 1)
        iter_data_time = time.time()

    # Update temperature. Done before saving so the stored value matches the
    # number of completed epochs recorded alongside it.
    model.update_temp()
    print(f'Update temperature to {model.temp}')

    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        save_all('latest', epoch)
        # Numbered snapshots are for evaluation, and resuming always goes
        # through 'latest', so they skip the (much larger) optimizer state
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, total_epoch, time.time() - epoch_start_time))
