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

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_cm_reg.py --component 205 --mode mr-ct --iteration 10003 --save_freq 500 --print_freq 200 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.0 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_cm_reg.py --component 205 --mode mr-ct --iteration 10003 --save_freq 500 --print_freq 200 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.0 --phase validate

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_cm_reg.py --component 205 --mode mr-ct --iteration 15003 --save_freq 500 --print_freq 200 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.0 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_cm_reg.py --component 205 --mode mr-ct --iteration 15003 --save_freq 500 --print_freq 200 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.0 --phase validate

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_cm_reg.py --component 205 --mode mr-ct --iteration 10003 --save_freq 500 --print_freq 200 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --phase train  --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_cm_reg.py --component 205 --mode mr-ct --iteration 10003 --save_freq 500 --print_freq 200 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --phase validate
