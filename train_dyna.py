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
options = TrainOptions()
opt = options.parse()

# Prepare the dataset
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Optionally hold images out to score the model each epoch. The held-out split
# uses the evaluation transform, since scoring on randomly augmented images
# would measure the augmentation as much as the model.
valset = None
if opt.val_size > 0:
    eval_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_source = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=eval_transform)
    cut = len(trainset) - opt.val_size
    valset = torch.utils.data.Subset(val_source, range(cut, len(val_source)))
    trainset = torch.utils.data.Subset(trainset, range(cut))
    print('#held-out images = %d (training on %d)' % (opt.val_size, cut))

dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=opt.num_workers, drop_last=True)
val_loader = None if valset is None else torch.utils.data.DataLoader(
    valset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)


# Create the checkpoint folder. opt.name is derived in BaseOptions.parse so that
# every script resolves the same run to the same folder.
path = os.path.join(opt.checkpoints_dir, opt.name)
if not os.path.exists(path):
    os.makedirs(path)
options.save_options(opt)      # provenance record next to the checkpoints

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


def save_all(tag, completed_epoch, extra=None):
    """Save the networks plus the state needed to resume from them."""
    state = {'epoch': completed_epoch, 'temp': model.temp, 'best_score': best_score}
    state.update(extra or {})
    model.save_networks(tag)
    model.save_training_state(tag, state)


@torch.no_grad()
def validate():
    """Score the model on the held-out split, lower is better.

    Uses the same objective that is being optimised, so a model is not called
    better for reconstructing well while transmitting more than it should.

    Two things make the score comparable across epochs rather than a lottery:
    evaluation runs at a fixed grid of SNRs instead of the random draw used in
    training, and the channel noise is drawn from a fixed seed. The RNG state is
    restored afterwards so scoring does not perturb the training stream.
    """
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(12345)

    was_training = opt.isTrain
    opt.isTrain = False        # fixed SNR, and the hard mask that inference uses
    model.eval()

    snrs = [opt.SNR_MIN, (opt.SNR_MIN + opt.SNR_MAX) // 2, opt.SNR_MAX]
    total, batches = 0.0, 0
    for images, _ in val_loader:
        model.set_input(images)
        for snr in snrs:
            model.opt.SNR = snr
            model.forward()
            mse = torch.mean((model.fake - model.real_B) ** 2).item()
            total += opt.lambda_L2 * mse + opt.lambda_reward * model.count.mean().item()
            batches += 1

    opt.isTrain = was_training
    model.train()
    torch.set_rng_state(rng_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return total / max(batches, 1)


# Pick up where a previous run stopped
best_score = float('inf')
start_epoch = opt.epoch_count
if opt.continue_train:
    state = model.load_training_state(opt.epoch)
    if state is None:
        print('WARNING: no training state alongside the checkpoint. Weights were '
              'loaded, but the optimizer and the temperature restart from scratch.')
    else:
        start_epoch = state['epoch'] + 1
        model.temp = state['temp']
        best_score = state.get('best_score', float('inf'))
        print('Resuming after epoch %d: starting at epoch %d with temperature %.5f'
              % (state['epoch'], start_epoch, model.temp))

    # The rolling save can be older than the best one, and would then carry a
    # worse best_score, letting a later mediocre epoch overwrite a better best.
    # The best checkpoint's own record is the authority.
    best_state_path = os.path.join(path, 'best_training_state.pth')
    if os.path.exists(best_state_path):
        recorded = torch.load(best_state_path, map_location='cpu',
                              weights_only=False).get('best_score')
        if recorded is not None and recorded < best_score:
            best_score = recorded
    if best_score < float('inf'):
        print('Best score so far: %.5f' % best_score)

if start_epoch > total_epoch:
    print('Nothing to do: epoch %d is already past the %d epoch schedule.'
          % (start_epoch, total_epoch))

current_stage = None

for epoch in range(start_epoch, total_epoch + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    epoch_loss, epoch_batches = 0.0, 0

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
        epoch_loss += model.loss_G.item()
        epoch_batches += 1

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

        if opt.save_latest_freq and total_iters % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            # This epoch is unfinished, so resume by repeating it from the start
            save_all(save_suffix, epoch - 1)
        iter_data_time = time.time()

    # Update temperature. Done before saving so the stored value matches the
    # number of completed epochs recorded alongside it.
    model.update_temp()
    print(f'Update temperature to {model.temp}')

    # Score the model and keep the best one separately from the rolling save.
    # Worth having because a run does not necessarily end at its best point:
    # variational circuits in particular can destabilise late in training.
    if epoch % opt.val_freq == 0:
        if val_loader is not None:
            score, label = validate(), 'val loss'
        else:
            score, label = epoch_loss / max(epoch_batches, 1), 'train loss'
        if score < best_score:
            best_score = score
            save_all('best', epoch, {'best_epoch': epoch})
            print('new best (%s %.5f at epoch %d), saved as best_*' % (label, score, epoch))
        else:
            print('%s %.5f (best %.5f)' % (label, score, best_score))

    # Overwrite the rolling resume point. Always on the final epoch too, so a
    # completed run does not leave 'latest' short of where it actually finished.
    if epoch % opt.save_epoch_freq == 0 or epoch == total_epoch:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        save_all('latest', epoch)

    if opt.snapshot_freq and epoch % opt.snapshot_freq == 0:
        # Numbered snapshots are for evaluation curves, and resuming always goes
        # through 'latest', so they skip the (much larger) optimizer state
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, total_epoch, time.time() - epoch_start_time))
