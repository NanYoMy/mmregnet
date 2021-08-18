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

/home/tens/wx518/anaconda3/envs/torch/bin/python ./vmx_SERN.py --phase train --task heart_ct_mr_Myo_96_cmspace_stand --subtask train_44unlabeled
/home/tens/wx518/anaconda3/envs/torch/bin/python ./vmx_SERN.py --phase train --task heart_ct_mr_Myo_96_cmspace_stand --subtask train_4labeled

/home/tens/wx518/anaconda3/envs/torch/bin/python ./vmx_SERN.py --phase train --task heart_ct_mr_Lv_96_cmspace_stand --subtask train_44unlabeled
/home/tens/wx518/anaconda3/envs/torch/bin/python ./vmx_SERN.py --phase train --task heart_ct_mr_Lv_96_cmspace_stand --subtask train_4labeled

