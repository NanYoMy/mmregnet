#!/bin/bash
#SBATCH -J conv_fusion
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 240:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname

##########mi
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 205 --Tatlas ct --Ttarget mr --patch_size 7
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 205 --Tatlas ct --Ttarget mr --patch_size 5
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 205 --Tatlas ct --Ttarget mr --patch_size 3
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 205 --Tatlas ct --Ttarget mr --patch_size 9


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 205 --Tatlas ct --Ttarget mr   --phase summary
