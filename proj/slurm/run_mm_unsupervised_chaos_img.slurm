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


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase train --batch_size 2 --fold 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase validate --batch_size 1 --fold 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase train --batch_size 2 --fold 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase validate --batch_size 1 --fold 2

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase train --batch_size 2 --fold 3
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase validate --batch_size 1 --fold 3

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase train --batch_size 2 --fold 4
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img.py --lr 0.001 --iteration 30003 --save_freq 500 --print_freq 200 --lambda_bend 1  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --task multi_chaos --component 1 --mode t1_in_DUAL_mr-t2SPIR_mr-ct --phase validate --batch_size 1 --fold 4
