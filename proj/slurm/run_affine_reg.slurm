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

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_abdomen.py --phase validate
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --phase validate

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 205 --components 205 --lambda_consis 0.01 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase train --batch_size 1
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 205 --components 205 --lambda_consis 0.01 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase validate --batch_size 1

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 500 --components 500 --lambda_consis 0.01 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase train --batch_size 1
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 500 --components 500 --lambda_consis 0.01 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase validate --batch_size 1

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase train --batch_size 1
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 205 --components 205 --lambda_consis 0.1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase validate --batch_size 1

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 205 --components 205 --lambda_consis 1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase train --batch_size 1
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./affinereg_mmwhs.py --fold 4 --lr 0.00001 --component 205 --components 205 --lambda_consis 1 --task MMWHS --Tatlas mr --Ttarget ct --iteration 15003 --save_freq 500 --print_freq 200 --phase validate --batch_size 1
