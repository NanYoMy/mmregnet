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


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 1 --task MMWHS --component 205 --mode mr-mr-ct --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 1 --task MMWHS --component 205 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 2 --task MMWHS --component 205 --mode mr-mr-ct  --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 2 --task MMWHS --component 205 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 3 --task MMWHS --component 205 --mode mr-mr-ct --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 3 --task MMWHS --component 205 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 4 --task MMWHS --component 205 --mode mr-mr-ct  --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 4 --task MMWHS --component 205 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 1 --task MMWHS --component 500 --mode mr-mr-ct --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 1 --task MMWHS --component 500 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 2 --task MMWHS --component 500 --mode mr-mr-ct  --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 2 --task MMWHS --component 500 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 3 --task MMWHS --component 500 --mode mr-mr-ct --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 3 --task MMWHS --component 500 --mode mr-mr-ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 4 --task MMWHS --component 500 --mode mr-mr-ct  --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./joint_unsupervised_mm_reg_img_gi.py  --iteration 15003 --save_freq 1000 --print_freq 500 --lr 0.001 --lambda_bend 10  --lambda_cycle_consis 0.1 --lambda_multi_consis 0.01 --fold 4 --task MMWHS --component 500 --mode mr-mr-ct --phase validate --batch_size 1
