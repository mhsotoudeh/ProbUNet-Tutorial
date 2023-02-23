#!/bin/bash

#SBATCH --time=0-10:00            # time (DD-HH:MM)

python -u "main.py"                                                                     \
\
--random_seed 0                                                                         \
\
--train_file "data/32_train.npy"                                                        \
--val_file "data/32_val.npy"                                                            \
--val_period 128       --val_bs 128                                                     \
\
--in_ch 1            --out_ch 1                                                         \
--intermediate_ch 16 32 64 64 64                                                       \
--scale_depth 1                                                                         \
--kernel_size 7 7 7 5 3                                                                 \
--padding_mode zeros                                                                    \
\
--latent_num 5                                                                          \
--latent_chs 1 1 1 1 1                                                                  \
--latent_locks 0 0 0 0 0                                                                \
\
--rec_type MSE                                                                          \
\
--loss_type ELBO        --beta 1.0                                                      \
\
--epochs 256            --bs 128                                                        \
--optimizer adamax                     --wd 1e-5                                        \
--lr 1e-4                              --scheduler_type cons                            \
\
--save_period 32                                                                        \
\
--output_dir "runs/part1"                                                               \
--comment "32_elbo"