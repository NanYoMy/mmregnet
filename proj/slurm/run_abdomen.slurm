#!/bin/bash
#SBATCH -J cascade
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 96:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_abdomen.py --phase train --lr 0.001 --lambda_ben 0.2 --lambda_consis 0.0 --iteration 2001
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_abdomen.py --phase validate --lr 0.001 --lambda_ben 0.2 --lambda_consis 0.0 --iteration 2001
