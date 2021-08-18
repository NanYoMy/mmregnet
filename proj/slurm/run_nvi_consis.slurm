#!/bin/bash
#SBATCH -J cascade
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 96:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./NVI_consis_reg.py --component 205 --components 205 --Tatlas mr --Ttarget ct --iteration 20003 --save_freq 500 --print_freq 200 --lambda_bend 10 --lambda_anti 0 --lambda_bi_consis 0 --lambda_multi_consis 0.1  --phase train
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./NVI_consis_reg.py --component 205 --components 205 --Tatlas mr --Ttarget ct --iteration 20003 --save_freq 500 --print_freq 200 --lambda_bend 10 --lambda_anti 0 --lambda_bi_consis 0 --lambda_multi_consis 0.1  --phase validate

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./NVI_consis_reg.py --component 205 --components 205 --Tatlas mr --Ttarget ct --iteration 20003 --save_freq 500 --print_freq 200 --lambda_bend 10 --lambda_anti 0 --lambda_bi_consis 0.1 --lambda_multi_consis 0.0  --phase train
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./NVI_consis_reg.py --component 205 --components 205 --Tatlas mr --Ttarget ct --iteration 20003 --save_freq 500 --print_freq 200 --lambda_bend 10 --lambda_anti 0 --lambda_bi_consis 0.1 --lambda_multi_consis 0.0 --phase validate

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./NVI_consis_reg.py --component 205 --components 205 --Tatlas mr --Ttarget ct --iteration 20003 --save_freq 500 --print_freq 200 --lambda_bend 10 --lambda_anti 0 --lambda_bi_consis 0.1 --lambda_multi_consis 0.1  --phase train
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./NVI_consis_reg.py --component 205 --components 205 --Tatlas mr --Ttarget ct --iteration 20003 --save_freq 500 --print_freq 200 --lambda_bend 10 --lambda_anti 0 --lambda_bi_consis 0.1 --lambda_multi_consis 0.1 --phase validate
