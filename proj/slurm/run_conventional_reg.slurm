#!/bin/bash
#SBATCH -J conv_reg
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 240:00:00

nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname

#elasticSyN
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type ElasticSyN --component 500 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type ElasticSyN --component 500 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type ElasticSyN --component 500 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 500 --Tatlas mr --Ttarget ct --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 500 --Tatlas mr --Ttarget ct --phase summary


##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type ElasticSyN --component 500 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type ElasticSyN --component 500 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type ElasticSyN --component 500 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 500 --Tatlas ct --Ttarget mr --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 500 --Tatlas ct --Ttarget mr --phase summary

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type ElasticSyN --component 205 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type ElasticSyN --component 205 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type ElasticSyN --component 205 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 205 --Tatlas ct --Ttarget mr --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 205 --Tatlas ct --Ttarget mr --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type ElasticSyN --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type ElasticSyN --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type ElasticSyN --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type ElasticSyN --component 1 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type ElasticSyN --component 1 --Tatlas ct --Ttarget mr --phase  summary

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type ElasticSyN --component 205 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type ElasticSyN --component 205 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type ElasticSyN --component 205 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 205 --Tatlas mr --Ttarget ct --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type ElasticSyN --component 205 --Tatlas mr --Ttarget ct --phase  summary

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type ElasticSyN --component 1 --Tatlas mr --Ttarget ct --phase reg
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type ElasticSyN --component 1 --Tatlas mr --Ttarget ct --phase reg
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type ElasticSyN --component 1 --Tatlas mr --Ttarget ct --phase reg
/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type ElasticSyN --component 1 --Tatlas mr --Ttarget ct --phase reg

/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type ElasticSyN --component 1 --Tatlas mr --Ttarget ct --phase  summary

#synonly
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNOnly --component 500 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNOnly --component 500 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNOnly --component 500 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 500 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 500 --Tatlas ct --Ttarget mr --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNOnly --component 205 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNOnly --component 205 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNOnly --component 205 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 205 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 205 --Tatlas ct --Ttarget mr --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type SyNOnly --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type SyNOnly --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type SyNOnly --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNOnly --component 1 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNOnly --component 1 --Tatlas ct --Ttarget mr --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNOnly --component 500 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNOnly --component 500 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNOnly --component 500 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 500 --Tatlas mr --Ttarget ct --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 500 --Tatlas mr --Ttarget ct --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNOnly --component 205 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNOnly --component 205 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNOnly --component 205 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 205 --Tatlas mr --Ttarget ct --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNOnly --component 205 --Tatlas mr --Ttarget ct --phase summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type SyNOnly --component 1 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type SyNOnly --component 1 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type SyNOnly --component 1 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNOnly --component 1 --Tatlas mr --Ttarget ct --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNOnly --component 1 --Tatlas mr --Ttarget ct --phase  summary


#syn
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyN --component 500 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyN --component 500 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyN --component 500 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyN --component 500 --Tatlas ct --Ttarget mr --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyN --component 205 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyN --component 205 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyN --component 205 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyN --component 205 --Tatlas ct --Ttarget mr --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type SyN --component 1 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type SyN --component 1 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type SyN --component 1 --Tatlas ct --Ttarget mr --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyN --component 1 --Tatlas ct --Ttarget mr --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyN --component 500 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyN --component 500 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyN --component 500 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyN --component 500 --Tatlas mr --Ttarget ct --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyN --component 205 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyN --component 205 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyN --component 205 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyN --component 205 --Tatlas mr --Ttarget ct --phase reg

##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type SyN --component 1 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type SyN --component 1 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type SyN --component 1 --Tatlas mr --Ttarget ct --phase reg
##/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyN --component 1 --Tatlas mr --Ttarget ct --phase reg

#syncc
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNCC --component 500 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNCC --component 500 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNCC --component 500 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 500 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 500 --Tatlas ct --Ttarget mr --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNCC --component 205 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNCC --component 205 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNCC --component 205 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 205 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 205 --Tatlas ct --Ttarget mr --phase summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type SyNCC --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type SyNCC --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type SyNCC --component 1 --Tatlas ct --Ttarget mr --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNCC --component 1 --Tatlas ct --Ttarget mr --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNCC --component 1 --Tatlas ct --Ttarget mr --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNCC --component 500 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNCC --component 500 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNCC --component 500 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 500 --Tatlas mr --Ttarget ct --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 500 --Tatlas mr --Ttarget ct --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 1 --type SyNCC --component 205 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 2 --type SyNCC --component 205 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 3 --type SyNCC --component 205 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 205 --Tatlas mr --Ttarget ct --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task MMWHS --fold 4 --type SyNCC --component 205 --Tatlas mr --Ttarget ct --phase  summary

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 1 --type SyNCC --component 1 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 2 --type SyNCC --component 1 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 3 --type SyNCC --component 1 --Tatlas mr --Ttarget ct --phase reg
#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNCC --component 1 --Tatlas mr --Ttarget ct --phase reg

#/home/tens/wx518/anaconda3/envs/dwb_tensorflow/bin/python ./convention_ant_reg.py --task CHAOS --fold 4 --type SyNCC --component 1 --Tatlas mr --Ttarget ct --phase  summary

