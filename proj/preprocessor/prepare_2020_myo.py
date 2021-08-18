import glob
import SimpleITK as sitk
from preprocessor.tools import get_bounding_box,crop_by_bbox,get_bounding_boxV2,sitkResize3DV2,sitkResample3DV2
import os
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_image
from dirutil.helper import mkdir_if_not_exist
from preprocessor.tools import reindex_label
def crop_by_label():
    '''
    :return:
    '''
    files = glob.glob("../datasets/myo_data/train25_myops_gd_convert/*.nii.gz")
    output_lab_dir="../datasets/myo_data/train25_myops_gd_crop/"
    output_img_dir="../datasets/myo_data/train25_myops_crop/"
    mkdir_if_not_exist(output_lab_dir)
    for i in files:
        lab = sitk.ReadImage(i)
        #先转化成统一space,保证crop的大小一致
        lab=sitkResample3DV2(lab,sitk.sitkNearestNeighbor,[1,1,1])
        bbox=get_bounding_boxV2(sitk.GetArrayFromImage(lab),padding=10)
        ##extend bbox
        crop_lab=crop_by_bbox(lab,bbox)
        crop_lab=sitkResize3DV2(crop_lab,[256,256,crop_lab.GetSize()[-1]],sitk.sitkNearestNeighbor)
        sitk_write_image(crop_lab,dir=output_lab_dir,name=os.path.basename(i))
        img_file=glob.glob("../datasets/myo_data/train25_convert/*%s*.nii.gz"%(os.path.basename(i).split("_")[2]))
        for j in img_file:
            img = sitk.ReadImage(j)
            img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
            crop_img=crop_by_bbox(img,bbox)
            crop_img = sitkResize3DV2(crop_img, [256, 256,crop_img.GetSize()[-1]], sitk.sitkLinear)
            sitk_write_image(crop_img, dir=output_img_dir, name=os.path.basename(j))

def slice_by_label():
    '''
    :return:
    '''
    files = glob.glob("../datasets/myo_data/train25_myops_gd_convert/*.nii.gz")
    output_lab_dir="../datasets/myo_data/train25_myops_gd_crop/"
    output_img_dir="../datasets/myo_data/train25_myops_crop/"
    mkdir_if_not_exist(output_lab_dir)
    for i in files:
        lab = sitk.ReadImage(i)
        #先转化成统一space,保证crop的大小一致
        lab=sitkResample3DV2(lab,sitk.sitkNearestNeighbor,[1,1,1])
        bbox=get_bounding_boxV2(sitk.GetArrayFromImage(lab),padding=10)
        ##extend bbox
        crop_lab=crop_by_bbox(lab,bbox)
        crop_lab=sitkResize3DV2(crop_lab,[256,256,crop_lab.GetSize()[-1]],sitk.sitkNearestNeighbor)
        sitk_write_image(crop_lab[:,:,crop_lab.GetSize()[-1]//2],dir=output_lab_dir,name=os.path.basename(i))
        img_file=glob.glob("../datasets/myo_data/train25_convert/*%s*.nii.gz"%(os.path.basename(i).split("_")[2]))
        for j in img_file:
            img = sitk.ReadImage(j)
            img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
            crop_img=crop_by_bbox(img,bbox)
            crop_img = sitkResize3DV2(crop_img, [256, 256,crop_img.GetSize()[-1]], sitk.sitkLinear)
            sitk_write_image(crop_img[:,:,crop_lab.GetSize()[-1]//2], dir=output_img_dir, name=os.path.basename(j))
