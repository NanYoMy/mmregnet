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

##### mv
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MV --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MV --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MV --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 205 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 205 --Tatlas ct --Ttarget mr --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MV --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MV --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MV --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 500 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 500 --Tatlas ct --Ttarget mr --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 1 --type MV --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 2 --type MV --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 3 --type MV --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MV --component 1 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MV --component 1 --Tatlas ct --Ttarget mr  --phase summary

#mr-ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MV --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MV --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MV --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 205 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 205 --Tatlas mr --Ttarget ct  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MV --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MV --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MV --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 500 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MV --component 500 --Tatlas mr --Ttarget ct  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 1 --type MV --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 2 --type MV --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 3 --type MV --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MV --component 1 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MV --component 1 --Tatlas mr --Ttarget ct   --phase summary



##########mi
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MI    --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MI    --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 205 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 205 --Tatlas ct --Ttarget mr   --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MI    --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MI    --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 500 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 500 --Tatlas ct --Ttarget mr  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 1 --type MI    --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 2 --type MI    --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 3 --type MI    --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MI    --component 1 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MI    --component 1 --Tatlas ct --Ttarget mr  --phase summary

#mr-ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MI    --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MI    --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 205 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 205 --Tatlas mr --Ttarget ct  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type MI    --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type MI    --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type MI    --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 500 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type MI    --component 500 --Tatlas mr --Ttarget ct  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 1 --type MI    --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 2 --type MI    --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 3 --type MI    --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MI    --component 1 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type MI    --component 1 --Tatlas mr --Ttarget ct  --phase summary

######staple
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type STAPLE --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type STAPLE --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type STAPLE --component 205 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 205 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 205 --Tatlas ct --Ttarget mr  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type STAPLE --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type STAPLE --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type STAPLE --component 500 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 500 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 500 --Tatlas ct --Ttarget mr  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 1 --type STAPLE --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 2 --type STAPLE --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 3 --type STAPLE --component 1 --Tatlas ct --Ttarget mr
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type STAPLE --component 1 --Tatlas ct --Ttarget mr

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type STAPLE --component 1 --Tatlas ct --Ttarget mr  --phase summary

#mr-ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type STAPLE --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type STAPLE --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type STAPLE --component 205 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 205 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 205 --Tatlas mr --Ttarget ct  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 1 --type STAPLE --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 2 --type STAPLE --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 3 --type STAPLE --component 500 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 500 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task MMWHS --fold 4 --type STAPLE --component 500 --Tatlas mr --Ttarget ct  --phase summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 1 --type STAPLE --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 2 --type STAPLE --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 3 --type STAPLE --component 1 --Tatlas mr --Ttarget ct
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type STAPLE --component 1 --Tatlas mr --Ttarget ct

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_fusion.py --task CHAOS --fold 4 --type STAPLE --component 1 --Tatlas mr --Ttarget ct  --phase summary


