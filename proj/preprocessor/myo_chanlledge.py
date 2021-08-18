import os
import glob
import SimpleITK as sitk
from sitkImageIO.itkdatawriter import sitk_wirte_ori_image


def convert_img():
    files=glob.glob("../../myo_data/train25/*.nii.gz")
    for i in files:
        img=sitk.GetArrayFromImage(sitk.ReadImage(i))
        sitk_wirte_ori_image(sitk.GetImageFromArray(img),'../../myo_data/train25_convert/',os.path.basename(i).split('.')[0])

def convert_lab():
    files = glob.glob("../../myo_data/train25_myops_gd/*.nii.gz")
    for i in files:
        img = sitk.GetArrayFromImage(sitk.ReadImage(i))
        sitk_wirte_ori_image(sitk.GetImageFromArray(img), '../../myo_data/train25_myops_gd_convert/',
                             os.path.basename(i).split('.')[0])
