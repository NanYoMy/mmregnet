#!/bin/bash
#SBATCH -J unet
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 240:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname
#1
##ct mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas ct --Ttarget mr --iteration 17003 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

##mr ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 1 --components 1 --task CHAOS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#205
##ct mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.001 --component 205 --components 205 --task MMWHS --Tatlas ct --Ttarget mr --iteration 10003 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#205
##mr ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 205 --components 205 --task MMWHS --Tatlas mr --Ttarget ct --iteration 10003 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#500
##ct mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.001 --component 500 --components 500 --task MMWHS --Tatlas ct --Ttarget mr --iteration 20003 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

#500
##mr ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 1 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 2 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 3 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 1000 --print_freq 200 --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./unet3d.py --fold 4 --lr 0.0001 --component 500 --components 500 --task MMWHS --Tatlas mr --Ttarget ct --iteration 17003 --save_freq 200 --print_freq 200 --phase summary --batch_size 1

