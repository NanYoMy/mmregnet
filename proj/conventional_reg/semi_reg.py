# -*- coding:utf-8 -*-
import os
from dirutil.helper import mkdir_if_not_exist,sort_glob,get_name_wo_suffix
import SimpleITK as sitk
import numpy as np

exe = "../zxhtool/zxhregsemi0.exe"
cmd="%s -target %s -source %s -rmask %s -o %s -Reg 3 -steps 50 50 20 -sub 4 4 4 -sub 4 4 4 -sub 2 2 2 -ffd 40 40 40 -ffd 20 20 20 -ffd 10 10 10 -bending 0.001"

def reg(args,p_target,p_target_lab,p_atlas,p_atlas_lab):
    cmd = r"%s -target %s -source %s -rmask %s -o %s -Reg 3 -steps 50 50 20 -sub 4 4 4 -sub 4 4 4 -sub 2 2 2 -ffd 40 40 40 -ffd 20 20 20 -ffd 10 10 10 -bending 0.001"\
          %(os.path.abspath(exe),p_target,p_atlas,p_atlas_lab,"%s/%s/%s"%(args.test_dir,"atlas_"+get_name_wo_suffix(p_atlas),get_name_wo_suffix(p_target_lab)))
    print(os.popen(cmd).read())

def to_int16(pathes):
    for path in pathes:
        img=sitk.ReadImage(path)
        img=sitk.Cast(img,sitk.sitkInt16)
        sitk.WriteImage(img,path)

def convert_type(args):
    atlas_imgs = sort_glob(args.dataset_dir + "/train_atlas/rez/img/*.nii.gz")
    target_imgs = sort_glob(args.dataset_dir + "/test_target/rez/img/*.nii.gz")
    to_int16(atlas_imgs)
    to_int16(target_imgs)

    atlas_labs = sort_glob(args.dataset_dir + "/train_atlas/rez/lab/*.nii.gz")
    target_labs = sort_glob(args.dataset_dir + "/test_target/rez/lab/*.nii.gz")
    to_int16(atlas_labs)
    to_int16(target_labs)


def registration_all_label_and_img(args,atlas_imgs, atlas_labs, target_imgs, target_labs):
    mkdir_if_not_exist(args.test_dir)
    for atlas_img,atlas_lab in zip(atlas_imgs,atlas_labs):
        mkdir_if_not_exist("%s/%s"%(args.test_dir,"atlas_"+get_name_wo_suffix(atlas_img)))
        mkdir_if_not_exist("%s/%s"%(args.test_dir,"atlas_"+get_name_wo_suffix(atlas_lab)))
        for target_img,target_lab in zip(target_imgs,target_labs):
            # reg(args,target_img,target_lab,atlas_img,atlas_lab)
            #测试
            reg(args,target_lab,target_lab,atlas_lab,atlas_lab)
            print("working:")
            print(atlas_img,atlas_lab,target_img,target_lab)

