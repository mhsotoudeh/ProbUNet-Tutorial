import argparse
import datetime
import json
import time
import socket
import tracemalloc
import os

import numpy as np
from torch.utils.data import DataLoader

from data import *
from model import *
from train import *

from torch.utils.tensorboard import SummaryWriter


# Define Arguments
parser = argparse.ArgumentParser(description="Hierarchical Probabilistic U-Net")

parser.add_argument("--random_seed", type=int, help="If provided, seed number will be set to the given value")


# Data

parser.add_argument("--train_file", help="<Required> Path to the Training Set", required=True)
parser.add_argument("--train_size", type=int, help="Training Set Size (If not provided, the whole examples in the traininig set will be used)")

parser.add_argument("--val_file", help="Path to the Validation Set")
parser.add_argument("--val_size", type=int, help="Validation Set Size (If not provided, the whole examples in the validation set will be used)")
parser.add_argument("--val_period", type=int, help="# Steps Between Consecutive Validations")
parser.add_argument("--val_bs", type=int, help="Validation Batch Size")

parser.add_argument("--normalization", help="Normalization Type (None/standard/log_normal)")


# Model

parser.add_argument("--in_ch", type=int, default=1, help="# Input Channels")
parser.add_argument("--out_ch", type=int, default=1, help="# Output Channels")
parser.add_argument("--intermediate_ch", type=int, nargs='+', help="<Required> Intermediate Channels", required=True)
parser.add_argument("--kernel_size", type=int, nargs='+', default=[3], help="Kernel Size of the Convolutional Layers at Each Scale")
parser.add_argument("--scale_depth", type=int, nargs='+', default=[1], help="Number of Residual Blocks at Each Scale")
parser.add_argument("--dilation", type=int, nargs='+', default=[1], help="Dilation at Each Scale")
parser.add_argument("--padding_mode", default='zeros', help="Padding Mode in the Decoder's Convolutional Layers")

parser.add_argument("--latent_num", type=int, default=0, help="Number of Latent Scales (Setting to zero results in a deterministic U-Net)")
parser.add_argument("--latent_chs", type=int, nargs='+', help="Number of Latent Channels at Each Latent Scale (Setting to None results in 1 channel per scale)")
parser.add_argument("--latent_locks", type=int, nargs='+', help="Whether Latent Space in Locked at Each Latent Scale (Setting to None makes all scales unlocked)")


# Loss

parser.add_argument("--rec_type", help="Reconstruction Loss Type (mse / loglikelihood)", required=True)
parser.add_argument("--ll_std_sample_num", type=int, help="(rec_type: loglikelihood) 0: Directly Predict Standard Devation / 1+: Number of Samples to Estimate Standard Deviation")

parser.add_argument("--k", type=float, help="If Provided, will use top-k Mask for Reconstruction Loss")
parser.add_argument("--topk_deterministic", action='store_true', help="Calculate top-k Mask Deterministically or Use Gumbell Trick")

parser.add_argument("--loss_type", default="ELBO", help="Loss Function Type (ELBO/GECO)")

parser.add_argument("--beta", type=float, default=1.0, help="(If Using ELBO Loss) Beta Parameter")
parser.add_argument("--beta_asc_steps", type=int, help="(If Using ELBO Loss with Beta Scheduler) Number of Ascending Steps (If Not Provided, Beta Will be Constant)")
parser.add_argument("--beta_cons_steps", type=int, default=1, help="(If Using ELBO Loss with Beta Scheduler) Number of Constant Steps")
parser.add_argument("--beta_saturation_step", type=int, help="(If Using ELBO Loss with Beta Scheduler) The Step at Which Beta Becomes Permanently 1")

parser.add_argument("--kappa", type=float, default=1.0, help="(If Using GECO Loss) Kappa Parameter")
parser.add_argument("--kappa_px", action='store_true', help="(If Using GECO Loss) Kappa Parameter Type (If true, Kappa should be provided per pixel)")
parser.add_argument("--decay", type=float, default=0.9, help="(If Using GECO Loss) EMA Decay Rate/Smoothing Factor")
parser.add_argument("--update_rate", type=float, default=0.01, help="(If Using GECO Loss) Lagrange Multiplier Update Rate")


# Training

parser.add_argument("--epochs", type=int, help="<Required> Number of Epochs", required=True)
parser.add_argument("--bs", type=int, help="<Required> Batch Size", required=True)

parser.add_argument("--optimizer", default="adam", help="Optimizer")
parser.add_argument("--wd", type=float, default=0.0, help="Weight Decay Parameter")

parser.add_argument("--lr", type=float, help="<Required> (Initial) Learning Rate", required=True)
parser.add_argument("--scheduler_type", default='cons', help="Scheduler Type (cons/step/milestones)")
parser.add_argument("--scheduler_step_size", type=int, default=128, help="Learning Rate Scheduler Step Size (If type is step)")
parser.add_argument("--scheduler_milestones", type=int, nargs='+', help="Learning Rate Scheduler Milestones (If type is milestones)")
parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Learning Rate Scheduler Gamma")

parser.add_argument("--save_period", type=int, default=128, help="Number of Epochs Between Saving the Model")

parser.add_argument("--output_dir", default="runs", help="Output Directory")
parser.add_argument("--comment", default="", help="Comment to be Included in the Stamp")


# Parse Arguments
args = parser.parse_args()


# Adjust Arguments
if args.latent_locks is None:
    args.latent_locks = [0] * args.latent_num
args.latent_locks = [bool(l) for l in args.latent_locks]

if len(args.kernel_size) < len(args.intermediate_ch):
    if len(args.kernel_size) == 1:
        args.kernel_size = args.kernel_size * len(args.intermediate_ch)
    else:
        print('Invalid kernel size, exiting...')
        exit()

if len(args.dilation) < len(args.intermediate_ch):
    if len(args.dilation) == 1:
        args.dilation = args.dilation * len(args.intermediate_ch)
    else:
        print('Invalid dilation, exiting...')
        exit()

if len(args.scale_depth) < len(args.intermediate_ch):
    if len(args.scale_depth) == 1:
        args.scale_depth = args.scale_depth * len(args.intermediate_ch)
    else:
        print('Invalid scale depth, exiting...')
        exit()


# Set Random Seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
else:
    np.random.seed(0)
    torch.manual_seed(0)


# Set Device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.available_gpus = torch.cuda.device_count()
device = torch.device(args.device)
print("Device is {}".format(device))


# Generate Stamp
# stamp = 'My Lovely HPUnet'  # Assign a name manually
timestamp = datetime.datetime.now().strftime('%m%d-%H%M')  # Assign a timestamp
compute_node = socket.gethostname()
suffix = datetime.datetime.now().strftime('%f')
stamp = timestamp + '_' + compute_node[:2] + '_' + suffix + '_' + args.comment
print('Stamp:', stamp)
args.compute_node = compute_node
args.stamp = stamp


# Initialize SummaryWriter (for tensorboard)
writer = SummaryWriter('{}/{}/tb'.format(args.output_dir, stamp))


# Load Data
train_data = prepare_data(args.train_file, size=args.train_size, normalization=args.normalization)[0]
train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
s = next(iter(train_data))[0].shape[-1]
args.size = s
args.pixels = s*s

val_file, val_loader = args.val_file, None
if val_file is not None:
    val_data = prepare_data(val_file, size=args.val_size, normalization=args.normalization)[0]
    val_loader = DataLoader(val_data, batch_size=args.val_bs, shuffle=False)


# Initialize Model
extra_out_ch = 1 if args.rec_type == 'loglikelihood' and args.ll_std_sample_num == 0 else 0

model = HPUNet( in_ch=args.in_ch, out_ch=args.out_ch+extra_out_ch, chs=args.intermediate_ch,
                latent_num=args.latent_num, latent_channels=args.latent_chs, latent_locks=args.latent_locks,
                scale_depth=args.scale_depth, kernel_size=args.kernel_size, dilation=args.dilation,
                padding_mode=args.padding_mode ).double()


args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


model.to(device)


# Set Loss Function

## Reconstruction Loss
if args.rec_type.lower() == 'mse':
    reconstruction_loss = MSELossWrapper()

elif args.rec_type.lower() == 'loglikelihood':
    reconstruction_loss = LogLikelihoodLoss()

else:
    print('Invalid reconstruction loss type, exiting...')
    exit()

## Masked Reconstruction
if args.k is not None:
    reconstruction_loss = TopkMaskedLoss(loss=reconstruction_loss, k=args.k, deterministic=args.topk_deterministic)

## Total Loss
if args.loss_type.lower() == 'elbo':
    if args.beta_asc_steps is None:
        beta_scheduler = BetaConstant(args.beta)
    else:
        beta_scheduler = BetaLinearScheduler(ascending_steps=args.beta_asc_steps, constant_steps=args.beta_cons_steps, max_beta=args.beta, saturation_step=args.beta_saturation_step)
    criterion = ELBOLoss(reconstruction_loss=reconstruction_loss, beta=beta_scheduler).to(device)

elif args.loss_type.lower() == 'geco':
    kappa = args.kappa
    if args.kappa_px is True:
        kappa *= args.pixels
    criterion = GECOLoss(reconstruction_loss=reconstruction_loss, kappa=kappa, decay=args.decay, update_rate=args.update_rate, device=device).to(device)

else:
    print('Invalid loss type, exiting...')
    exit()


# Set Optimizer
if args.optimizer == 'adamax':
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wd)

elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

else:
    print('Optimizer not known, exiting...')
    exit()


# Set LR Scheduler
if args.scheduler_type == 'cons':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs)

elif args.scheduler_type == 'step':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

elif args.scheduler_type == 'milestones':
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)


# Save Args
argsdict = vars(args)
with open('{}/{}/args.json'.format(args.output_dir, stamp), 'w') as f:
    json.dump(argsdict, f)
with open('{}/{}/args.txt'.format(args.output_dir, stamp), 'w') as f:
    for k in argsdict.keys():
        f.write("'{}': '{}'\n".format(k, argsdict[k]))


# Start Timing
start = time.time()

# Train the Model
history = train_model(args, model, train_loader, criterion, optimizer, lr_scheduler, writer, device, val_loader, start)


# End Timing & Report Training Time
end = time.time()
training_time = (end - start) / 3600
history['training_time(hours)'] = training_time
print('Training done in {:.1f} hours'.format(training_time))


# Save Model, Loss and History
torch.save(model, '{}/{}/model.pth'.format(args.output_dir, stamp))

torch.save(criterion, '{}/{}/loss.pth'.format(args.output_dir, stamp))

with open('{}/{}/history.json'.format(args.output_dir, stamp), 'w') as f:
    json.dump(history, f)