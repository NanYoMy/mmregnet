#!/bin/bash
#SBATCH -J  labreg
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 240:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas mr --Ttarget ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py  --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas mr --Ttarget ct --phase summary --batch_size 1


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas mr --Ttarget ct --phase summary --batch_size 1


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 1 --component 1 --lambda_ben 0.5 --task CHAOS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas ct --Ttarget mr --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 1 --component 1 --task CHAOS --Tatlas ct --Ttarget mr --phase summary --batch_size 1


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 500 --component 500 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 500 --component 500 --task MMWHS --Tatlas ct --Ttarget mr --phase summary --batch_size 1


/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas ct --Ttarget mr --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas ct --Ttarget mr --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas ct --Ttarget mr --phase summary --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 1 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 2 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 3 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 205 --component 205 --lambda_ben 0.5 --task MMWHS --Tatlas mr --Ttarget ct --iteration 1503 --save_freq 500 --print_freq 200 --phase train --batch_size 2
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas mr --Ttarget ct --phase validate --batch_size 1

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./learn2reg_OneDDFLabReg.py --fold 4 --lr 0.001 --components 205 --component 205 --task MMWHS --Tatlas mr --Ttarget ct --phase summary --batch_size 1
