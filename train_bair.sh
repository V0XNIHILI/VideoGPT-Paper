CUDA_VISIBLE_DEVICES=4,5 python train_videogpt.py -o ./bair_outputs --cfg gpt_small --vqvae_ckpt bair_stride4x2x2 --dataset bair_pushing --n_cond_frames 3 --batch_size 96 --total_iter 150000