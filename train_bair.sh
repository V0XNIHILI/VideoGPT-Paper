CUDA_VISIBLE_DEVICES=5 python train_videogpt.py -o ./bair_outputs --cfg gpt_small --vqvae_ckpt /space/ddenblanken/Projects/VideoGPT/bair_stride4x2x2 --dataset bair_pushing --n_cond_frames 1 --batch_size 4 --total_iter 150000