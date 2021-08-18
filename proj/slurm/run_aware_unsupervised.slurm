#!/bin/bash
#SBATCH -J nvireg
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 96:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname



/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./aware_unsupervised_cm_reg.py  --iteration 40003 --save_freq 500 --print_freq 200 --lr 0.0001 --lambda_bend 10  --lambda_cycle_consis 0.1  --lambda_aware 0.2  --batch_size 2 --phase train
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./aware_unsupervised_cm_reg.py  --iteration 40003 --save_freq 500 --print_freq 200 --lr 0.0001 --lambda_bend 10  --lambda_cycle_consis 0.1  --lambda_aware 0.2  --batch_size 2 --phase validate

