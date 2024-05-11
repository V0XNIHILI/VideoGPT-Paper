#!/bin/bash
#TODO: add cluster settings

#TODO: install and activate environment
#

python train_videogpt.py -o TODO --cfg gpt_large --vqvae_ckpt TODO --dataset ucf101 --n_cond_frames 0 --batch_size 32 --total_iter 200000

# -o : output directory of the training
# --vqvae_ckpt: path to the downloaded pretrained VQVAE
#
# there are other TODO in the rest of the repo you might need to look at
