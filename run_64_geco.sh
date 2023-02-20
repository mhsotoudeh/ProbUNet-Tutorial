#!/bin/bash

#SBATCH --account=rrg-lplevass
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=10G                 # memory (per node)
#SBATCH --time=1-00:00            # time (DD-HH:MM)

python -u "main.py"                                                              \
\
--random_seed 0                                                                         \
\
--train_file "utils/32768-12288-2899/64_train.npy"                                      \
--val_file "utils/32768-12288-2899/64_val.npy"                                          \
--val_period 128       --val_bs 128                                                     \
\
--in_ch 1            --out_ch 1                                                         \
--intermediate_ch 16 16 32 64 64 64                                                        \
--scale_depth 1                                                                         \
--kernel_size 7 7 7 7 5 3                                                                 \
--padding_mode zeros                                                                    \
\
--latent_num 6                                                                          \
--latent_chs 1 1 1 1 1 1                                                                  \
--latent_locks 0 0 0 0 0 0                                                                \
\
--rec_type MSE                                                                          \
\
--loss_type GECO     --kappa_px        --kappa 6e-5                                      \
--decay 0.9          --update_rate 10.0                                                  \
\
--epochs 256            --bs 128                                                           \
--optimizer adamax                     --wd 1e-5                                        \
--lr 1e-4                              --scheduler_type cons                            \
\
--save_period 32                                                                        \
\
--output_dir "runs/part1/64"                                                                 \
--comment "lensing_64_geco_kappa6e-5_update10.0"