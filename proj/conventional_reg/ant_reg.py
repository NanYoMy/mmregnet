# import ants
from sitkImageIO.itkdatawriter import sitk_write_lab,sitk_write_image
import numpy as np
import SimpleITK as sitk
import os
from dirutil.helper import mkdir_if_not_exist
from dirutil.helper import sort_glob
from preprocessor.tools import rescale_one_dir
from evaluate.metric import calculate_binary_hd,calculate_binary_dice,print_mean_and_std
from dirutil.helper import get_name_wo_suffix
from excelutil.output2excel import outpu2excel
from dirutil.helper import sort_glob
from learn2reg.sampler import MMSampler


# def registration_all_label_and_img(args,atlas_imgs, atlas_labs, target_imgs, target_labs):
#
#     res=[]
#     for target_img,target_lab in zip(target_imgs,target_labs):
#         for atlas_img,atlas_lab in zip(atlas_imgs,atlas_labs):
#             print("working:")
#             print(atlas_img,atlas_lab,target_img,target_lab)
#             reg(args,target_img,target_lab,atlas_img,atlas_lab)


'''
把数据转化成0,255之间
'''
def rescale(args):
    atlas_imgs=sort_glob(args.dataset_dir+"/train_atlas/rez/img/*.nii.gz")
    target_imgs=sort_glob(args.dataset_dir+"/validate_target/rez/img/*.nii.gz")
    target_imgs=target_imgs+sort_glob(args.dataset_dir+"/train_target/rez/img/*.nii.gz")
    rescale_one_dir(atlas_imgs)
    rescale_one_dir(target_imgs)
from learn2reg.sampler import Sampler
class AntReg():
    def __init__(self,args):
        self.args=args
        self.train_sampler = Sampler(self.args, 'train')
        self.validate_sampler = Sampler(self.args, 'validate')

    def run_reg(self):
        all_ds=[]
        all_hd=[]
        for target_img, target_lab in zip(self.validate_sampler.img_fix, self.validate_sampler.lab_fix):
            for atlas_img, atlas_lab in zip(self.validate_sampler.img_mv, self.validate_sampler.lab_mv):
                print("working:")
                print(atlas_img, atlas_lab, target_img, target_lab)
                ds,hd= self.reg_one_pair(target_img, target_lab, atlas_img, atlas_lab)
                all_ds.append(ds)
                all_hd.append(hd)
                print("ds %f  hd %f"%(ds,hd))
        print_mean_and_std(all_ds)
        print_mean_and_std(all_hd)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_DS",all_ds)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_HD",all_hd)

    def reg_one_pair(self, fix_img_path, fix_label_path, move_img_path, move_label_path):
        type = self.args.type
        # 读取数据，格式为： ants.core.ants_image.ANTsImage
        fix_img = ants.image_read(fix_img_path)
        fix_label = ants.image_read(fix_label_path)
        move_img = ants.image_read(move_img_path)
        move_label = ants.image_read(move_label_path)

        g1 = ants.iMath_grad(fix_img)
        g2 = ants.iMath_grad(move_img)
        demonsMetric = ['demons', g1, g2, 1, 1]
        ccMetric = ['CC', fix_img, move_img, 2, 4]
        metrics = list()
        metrics.append(demonsMetric)
        # 配准
        # outs = ants.registration(fix_img,move_img,type_of_transforme = 'Affine')
        # outs = ants.registration( fix_img, move_img, 'ElasticSyN',  multivariate_extras = metrics )
        # outs = ants.registration( fix_img, move_img, type,syn_metric='demons' )
        # outs = ants.registration( fix_img, move_img, type,verbose=True)
        fix_mask=fix_img>fix_img.mean()
        outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform=type,mask=fix_mask, reg_iterations=(20, 20, 40))
        # 获取配准后的数据，并保存
        # ants.image_write(outs['warpedmovout']  ,'./warp_image.nii.gz')
        print(outs)
        if len(outs['fwdtransforms']) != 2:
            # return [0]
            print("invalid output")
        # 获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
        warp_label = ants.apply_transforms(fix_img, move_label, transformlist=outs['fwdtransforms'],interpolator='nearestNeighbor')
        warp_img= ants.apply_transforms(fix_img, move_img, transformlist=outs['fwdtransforms'],interpolator='nearestNeighbor')
        out_dir = self.args.sample_dir + "/target_"+get_name_wo_suffix(fix_img_path)
        mkdir_if_not_exist(out_dir)

        p_warp_mv_label = out_dir + "/" + os.path.basename(move_label_path)
        ants.image_write(warp_label, p_warp_mv_label)
        p_warp_mv_img= out_dir + "/" + os.path.basename(move_img_path)
        ants.image_write(warp_img, p_warp_mv_img)

        p_fix_label = out_dir + "/" + os.path.basename(fix_label_path)
        ants.image_write(fix_label, p_fix_label)
        p_fix_img= out_dir + "/" + os.path.basename(fix_img_path)
        ants.image_write(fix_img, p_fix_img)




        fix_label=sitk.ReadImage(p_fix_label)
        fix_label_array=np.where(sitk.GetArrayFromImage(fix_label)==self.args.component,1,0)
        sitk_write_lab(fix_label_array,fix_label,out_dir,get_name_wo_suffix(p_fix_label))

        warp_mv_label=sitk.ReadImage(p_warp_mv_label)
        warp_mv_label_array=np.where(sitk.GetArrayFromImage(warp_mv_label)==self.args.component,1,0)
        sitk_write_lab(warp_mv_label_array,warp_mv_label,out_dir,get_name_wo_suffix(p_warp_mv_label))

        ds=calculate_binary_dice(fix_label_array,warp_mv_label_array)
        hd=calculate_binary_hd(fix_label_array,warp_mv_label_array,spacing=fix_label.GetSpacing())
        return ds,hd

    def reg_one_pairV2(self, fix_img_path, fix_label_path, move_img_path, move_label_path):
        def command_iteration(method):
            print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                                   method.GetMetricValue(),
                                                   method.GetOptimizerPosition()))

        def command_multi_iteration(method):
            print("--------- Resolution Changing ---------")

        fixed = sitk.ReadImage(fix_img_path, sitk.sitkFloat32)
        fixed = sitk.Normalize(fixed)
        fixed = sitk.DiscreteGaussian(fixed, 2.0)

        fixed_lab = sitk.ReadImage(fix_label_path, sitk.sitkUInt16)

        moving = sitk.ReadImage(move_img_path, sitk.sitkFloat32)
        moving = sitk.Normalize(moving)
        moving = sitk.DiscreteGaussian(moving, 2.0)

        moving_lab = sitk.ReadImage(move_label_path, sitk.sitkFloat32)

        transformDomainMeshSize = [10] * moving.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed,
                                              transformDomainMeshSize)

        print("Initial Parameters:")
        print(tx.GetParameters())

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
                                                  convergenceMinimumValue=1e-4,
                                                  convergenceWindowSize=5)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(tx)
        R.SetInterpolator(sitk.sitkLinear)

        R.SetShrinkFactorsPerLevel([6, 2, 1])
        R.SetSmoothingSigmasPerLevel([6, 2, 1])

        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: command_multi_iteration(R))

        outTx = R.Execute(fixed, moving)
        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}"
              .format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

        # sitk.WriteTransform(outTx,'../outputs/tmp.nii.gz' )

        if True:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(outTx)

            warp_mv_img= resampler.Execute(moving)
            warp_mv_label= resampler.Execute(moving_lab)
            warp_mv_label= sitk.Cast(warp_mv_label, sitk.sitkUInt16)

            # cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
            # sitk.Show(cimg, "ImageRegistration4 Composition")


        # out_dir = self.args.sample_dir + "/target_"+get_name_wo_suffix(fix_img_path)
        out_dir = "../outputs/tmp/"
        mkdir_if_not_exist(out_dir)

        sitk_write_image(warp_mv_img,fixed,out_dir,get_name_wo_suffix(move_img_path))
        sitk_write_image(fixed,fixed,out_dir,get_name_wo_suffix(fix_img_path))

        fix_label_array=np.where(sitk.GetArrayFromImage(fixed_lab)==self.args.component,1,0)
        sitk_write_lab(fix_label_array,fixed_lab,out_dir,get_name_wo_suffix(fix_label_path))

        warp_mv_label_array=np.where(sitk.GetArrayFromImage(warp_mv_label)==self.args.component,1,0)
        sitk_write_lab(warp_mv_label_array,warp_mv_label,out_dir,get_name_wo_suffix(move_label_path))

        ds=calculate_binary_dice(fix_label_array,warp_mv_label_array)
        hd=calculate_binary_hd(fix_label_array,warp_mv_label_array,spacing=fixed_lab.GetSpacing())
        return ds,hd


    def validate(self):
        dirs=sort_glob(self.args.sample_dir+"/*")
        DS=[]
        HD=[]
        for d in dirs:
            target_file=sort_glob(d+"/*%s*label*"%self.args.Ttarget)
            atlas_file=sort_glob(d+"/*%s*label*"%self.args.Tatlas)
            fix_label=sitk.ReadImage(target_file[0])
            fix_array=sitk.GetArrayFromImage(fix_label)
            for itr in atlas_file:
                mv_img=sitk.ReadImage(itr)
                mv_array=sitk.GetArrayFromImage(mv_img)
                ds=calculate_binary_dice(fix_array,mv_array)
                hd=calculate_binary_hd(fix_array,mv_array,spacing=fix_label.GetSpacing())
                print(ds)
                DS.append(ds)
                HD.append(hd)
        outpu2excel(self.args.res_excel, self.args.MOLD_ID + "_DS", DS)
        outpu2excel(self.args.res_excel, self.args.MOLD_ID + "_HD", HD)


class AntRegV2():
    def __init__(self,args):
        self.args=args
        self.train_sampler = Sampler(self.args, 'train')
        self.validate_sampler = MMSampler(self.args, 'validate')

    def run_reg(self):
        all_ds=[]
        all_hd=[]
        for target_img, target_lab in zip(self.validate_sampler.img_fix, self.validate_sampler.lab_fix):
            for atlas_img, atlas_lab in zip(self.validate_sampler.img_mv1, self.validate_sampler.lab_mv1):
                print("working:")
                print(atlas_img, atlas_lab, target_img, target_lab)
                ds,hd= self.reg_one_pair(target_img, target_lab, atlas_img, atlas_lab)
                all_ds.append(ds)
                all_hd.append(hd)
                print("ds %f  hd %f"%(ds,hd))
        print_mean_and_std(all_ds)
        print_mean_and_std(all_hd)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_DS",all_ds)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_HD",all_hd)

    def reg_one_pair(self, fix_img_path, fix_label_path, move_img_path, move_label_path):
        type = self.args.type
        # 读取数据，格式为： ants.core.ants_image.ANTsImage
        fix_img = ants.image_read(fix_img_path)
        fix_label = ants.image_read(fix_label_path)
        move_img = ants.image_read(move_img_path)
        move_label = ants.image_read(move_label_path)

        g1 = ants.iMath_grad(fix_img)
        g2 = ants.iMath_grad(move_img)
        demonsMetric = ['demons', g1, g2, 1, 1]
        ccMetric = ['CC', fix_img, move_img, 2, 4]
        metrics = list()
        metrics.append(demonsMetric)
        # 配准
        # outs = ants.registration(fix_img,move_img,type_of_transforme = 'Affine')
        # outs = ants.registration( fix_img, move_img, 'ElasticSyN',  multivariate_extras = metrics )
        # outs = ants.registration( fix_img, move_img, type,syn_metric='demons' )
        # outs = ants.registration( fix_img, move_img, type,verbose=True)
        outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform=type, reg_iterations=(20, 20, 40))
        # 获取配准后的数据，并保存
        # ants.image_write(outs['warpedmovout']  ,'./warp_image.nii.gz')
        print(outs)
        if len(outs['fwdtransforms']) != 2:
            # return [0]
            print("invalid output")
        # 获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
        warp_label = ants.apply_transforms(fix_img, move_label, transformlist=outs['fwdtransforms'],interpolator='nearestNeighbor')
        warp_img= ants.apply_transforms(fix_img, move_img, transformlist=outs['fwdtransforms'],interpolator='nearestNeighbor')
        out_dir = self.args.sample_dir + "/target_"+get_name_wo_suffix(fix_img_path)
        mkdir_if_not_exist(out_dir)

        p_warp_mv_label = out_dir + "/" + os.path.basename(move_label_path)
        ants.image_write(warp_label, p_warp_mv_label)
        p_warp_mv_img= out_dir + "/" + os.path.basename(move_img_path)
        ants.image_write(warp_img, p_warp_mv_img)

        p_fix_label = out_dir + "/" + os.path.basename(fix_label_path)
        ants.image_write(fix_label, p_fix_label)
        p_fix_img= out_dir + "/" + os.path.basename(fix_img_path)
        ants.image_write(fix_img, p_fix_img)


        fix_label=sitk.ReadImage(p_fix_label)
        fix_label_array=np.where(sitk.GetArrayFromImage(fix_label)==self.args.component,1,0)
        sitk_write_lab(fix_label_array,fix_label,out_dir,get_name_wo_suffix(p_fix_label))

        warp_mv_label=sitk.ReadImage(p_warp_mv_label)
        warp_mv_label_array=np.where(sitk.GetArrayFromImage(warp_mv_label)==self.args.component,1,0)
        sitk_write_lab(warp_mv_label_array,warp_mv_label,out_dir,get_name_wo_suffix(p_warp_mv_label))

        ds=calculate_binary_dice(fix_label_array,warp_mv_label_array)
        hd=calculate_binary_hd(fix_label_array,warp_mv_label_array,spacing=fix_label.GetSpacing())
        return ds,hd

    def reg_one_pairV2(self, fix_img_path, fix_label_path, move_img_path, move_label_path):
        def command_iteration(method):
            print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                                   method.GetMetricValue(),
                                                   method.GetOptimizerPosition()))

        def command_multi_iteration(method):
            print("--------- Resolution Changing ---------")

        fixed = sitk.ReadImage(fix_img_path, sitk.sitkFloat32)
        fixed = sitk.Normalize(fixed)
        fixed = sitk.DiscreteGaussian(fixed, 2.0)

        fixed_lab = sitk.ReadImage(fix_label_path, sitk.sitkUInt16)

        moving = sitk.ReadImage(move_img_path, sitk.sitkFloat32)
        moving = sitk.Normalize(moving)
        moving = sitk.DiscreteGaussian(moving, 2.0)

        moving_lab = sitk.ReadImage(move_label_path, sitk.sitkFloat32)

        transformDomainMeshSize = [10] * moving.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed,
                                              transformDomainMeshSize)

        print("Initial Parameters:")
        print(tx.GetParameters())

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
                                                  convergenceMinimumValue=1e-4,
                                                  convergenceWindowSize=5)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(tx)
        R.SetInterpolator(sitk.sitkLinear)

        R.SetShrinkFactorsPerLevel([6, 2, 1])
        R.SetSmoothingSigmasPerLevel([6, 2, 1])

        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: command_multi_iteration(R))

        outTx = R.Execute(fixed, moving)
        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}"
              .format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

        # sitk.WriteTransform(outTx,'../outputs/tmp.nii.gz' )

        if True:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(outTx)

            warp_mv_img= resampler.Execute(moving)
            warp_mv_label= resampler.Execute(moving_lab)
            warp_mv_label= sitk.Cast(warp_mv_label, sitk.sitkUInt16)

            # cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
            # sitk.Show(cimg, "ImageRegistration4 Composition")


        # out_dir = self.args.sample_dir + "/target_"+get_name_wo_suffix(fix_img_path)
        out_dir = "../outputs/tmp/"
        mkdir_if_not_exist(out_dir)

        sitk_write_image(warp_mv_img,fixed,out_dir,get_name_wo_suffix(move_img_path))
        sitk_write_image(fixed,fixed,out_dir,get_name_wo_suffix(fix_img_path))

        fix_label_array=np.where(sitk.GetArrayFromImage(fixed_lab)==self.args.component,1,0)
        sitk_write_lab(fix_label_array,fixed_lab,out_dir,get_name_wo_suffix(fix_label_path))

        warp_mv_label_array=np.where(sitk.GetArrayFromImage(warp_mv_label)==self.args.component,1,0)
        sitk_write_lab(warp_mv_label_array,warp_mv_label,out_dir,get_name_wo_suffix(move_label_path))

        ds=calculate_binary_dice(fix_label_array,warp_mv_label_array)
        hd=calculate_binary_hd(fix_label_array,warp_mv_label_array,spacing=fixed_lab.GetSpacing())
        return ds,hd


    def validate(self):
        dirs=sort_glob(self.args.sample_dir+"/*")
        DS=[]
        HD=[]
        for d in dirs:
            target_file=sort_glob(d+"/*%s*label*"%self.args.Ttarget)
            atlas_file=sort_glob(d+"/*%s*label*"%self.args.Tatlas)
            fix_label=sitk.ReadImage(target_file[0])
            fix_array=sitk.GetArrayFromImage(fix_label)
            for itr in atlas_file:
                mv_img=sitk.ReadImage(itr)
                mv_array=sitk.GetArrayFromImage(mv_img)
                ds=calculate_binary_dice(fix_array,mv_array)
                hd=calculate_binary_hd(fix_array,mv_array,spacing=fix_label.GetSpacing())
                print(ds)
                DS.append(ds)
                HD.append(hd)
        outpu2excel(self.args.res_excel, self.args.MOLD_ID + "_DS", DS)
        outpu2excel(self.args.res_excel, self.args.MOLD_ID + "_HD", HD)
# import ants
from dirutil.helper import mkdir_if_not_exist
class AntRegV3():
    def __init__(self,args):
        self.args=args
        self.train_sampler = Sampler(self.args, 'train')
        self.validate_sampler = MMSampler(self.args, 'validate')

    def run_reg(self,dir):

        fix_imgs=sort_glob(dir+"/*fixe_img*")
        rescale_one_dir(fix_imgs)
        fix_labs = sort_glob(dir + "/*fixe_lab*")
        mv_imgs = sort_glob(dir + "/*input_mv*img*")
        rescale_one_dir(mv_imgs)
        mv_labs = sort_glob(dir + "/*input_mv*lab*")

        all_ds=[]
        all_hd=[]

        for atlas_img, atlas_lab  in zip(mv_imgs,mv_labs):
            for target_img, target_lab in zip(fix_imgs,fix_labs):
                print("working:")
                print(atlas_img, atlas_lab, target_img, target_lab)
                mkdir_if_not_exist(dir+"/%s"%self.args.type)
                ds,hd= self.reg_one_pair(target_img, target_lab, atlas_img, atlas_lab,dir+"/%s"%self.args.type)
                all_ds.append(ds)
                all_hd.append(hd)
                print("ds %f  hd %f"%(ds,hd))
        print_mean_and_std(all_ds)
        print_mean_and_std(all_hd)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_DS",all_ds)
        outpu2excel(self.args.res_excel,self.args.MOLD_ID+"_HD",all_hd)

    def reg_one_pair(self, fix_img_path, fix_label_path, move_img_path, move_label_path,out_dir):
        type = self.args.type
        # 读取数据，格式为： ants.core.ants_image.ANTsImage
        fix_img = ants.image_read(fix_img_path)
        fix_label = ants.image_read(fix_label_path)
        move_img = ants.image_read(move_img_path)
        move_label = ants.image_read(move_label_path)

        g1 = ants.iMath_grad(fix_img)
        g2 = ants.iMath_grad(move_img)
        demonsMetric = ['demons', g1, g2, 1, 1]
        ccMetric = ['CC', fix_img, move_img, 2, 4]
        metrics = list()
        metrics.append(demonsMetric)
        # 配准
        # outs = ants.registration(fix_img,move_img,type_of_transforme = 'Affine')
        # outs = ants.registration( fix_img, move_img, 'ElasticSyN',  multivariate_extras = metrics )
        # outs = ants.registration( fix_img, move_img, type,syn_metric='demons' )
        # outs = ants.registration( fix_img, move_img, type,verbose=True)
        outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform=type, reg_iterations=(20, 20, 40))
        # 获取配准后的数据，并保存
        # ants.image_write(outs['warpedmovout']  ,'./warp_image.nii.gz')
        print(outs)
        if len(outs['fwdtransforms']) != 2:
            # return [0]
            print("invalid output")
        # 获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
        warp_label = ants.apply_transforms(fix_img, move_label, transformlist=outs['fwdtransforms'],interpolator='nearestNeighbor')
        warp_img= ants.apply_transforms(fix_img, move_img, transformlist=outs['fwdtransforms'],interpolator='nearestNeighbor')

        mkdir_if_not_exist(out_dir)

        p_warp_mv_label = out_dir + "/" + os.path.basename(move_label_path)
        ants.image_write(warp_label, p_warp_mv_label)
        p_warp_mv_img= out_dir + "/" + os.path.basename(move_img_path)
        ants.image_write(warp_img, p_warp_mv_img)

        p_fix_label = out_dir + "/" + os.path.basename(fix_label_path)
        ants.image_write(fix_label, p_fix_label)
        p_fix_img= out_dir + "/" + os.path.basename(fix_img_path)
        ants.image_write(fix_img, p_fix_img)


        fix_label=sitk.ReadImage(p_fix_label)
        fix_label_array=np.where(sitk.GetArrayFromImage(fix_label)==self.args.component,1,0)
        sitk_write_lab(fix_label_array,fix_label,out_dir,get_name_wo_suffix(p_fix_label))

        warp_mv_label=sitk.ReadImage(p_warp_mv_label)
        warp_mv_label_array=np.where(sitk.GetArrayFromImage(warp_mv_label)==self.args.component,1,0)
        sitk_write_lab(warp_mv_label_array,warp_mv_label,out_dir,get_name_wo_suffix(p_warp_mv_label))

        ds=calculate_binary_dice(fix_label_array,warp_mv_label_array)
        hd=calculate_binary_hd(fix_label_array,warp_mv_label_array,spacing=fix_label.GetSpacing())
        return ds,hd

    def reg_one_pairV2(self, fix_img_path, fix_label_path, move_img_path, move_label_path):
        def command_iteration(method):
            print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                                   method.GetMetricValue(),
                                                   method.GetOptimizerPosition()))

        def command_multi_iteration(method):
            print("--------- Resolution Changing ---------")

        fixed = sitk.ReadImage(fix_img_path, sitk.sitkFloat32)
        fixed = sitk.Normalize(fixed)
        fixed = sitk.DiscreteGaussian(fixed, 2.0)

        fixed_lab = sitk.ReadImage(fix_label_path, sitk.sitkUInt16)

        moving = sitk.ReadImage(move_img_path, sitk.sitkFloat32)
        moving = sitk.Normalize(moving)
        moving = sitk.DiscreteGaussian(moving, 2.0)

        moving_lab = sitk.ReadImage(move_label_path, sitk.sitkFloat32)

        transformDomainMeshSize = [10] * moving.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed,
                                              transformDomainMeshSize)

        print("Initial Parameters:")
        print(tx.GetParameters())

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
                                                  convergenceMinimumValue=1e-4,
                                                  convergenceWindowSize=5)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(tx)
        R.SetInterpolator(sitk.sitkLinear)

        R.SetShrinkFactorsPerLevel([6, 2, 1])
        R.SetSmoothingSigmasPerLevel([6, 2, 1])

        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: command_multi_iteration(R))

        outTx = R.Execute(fixed, moving)
        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}"
              .format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

        # sitk.WriteTransform(outTx,'../outputs/tmp.nii.gz' )

        if True:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(outTx)

            warp_mv_img= resampler.Execute(moving)
            warp_mv_label= resampler.Execute(moving_lab)
            warp_mv_label= sitk.Cast(warp_mv_label, sitk.sitkUInt16)

            # cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
            # sitk.Show(cimg, "ImageRegistration4 Composition")


        # out_dir = self.args.sample_dir + "/target_"+get_name_wo_suffix(fix_img_path)
        out_dir = "../outputs/tmp/"
        mkdir_if_not_exist(out_dir)

        sitk_write_image(warp_mv_img,fixed,out_dir,get_name_wo_suffix(move_img_path))
        sitk_write_image(fixed,fixed,out_dir,get_name_wo_suffix(fix_img_path))

        fix_label_array=np.where(sitk.GetArrayFromImage(fixed_lab)==self.args.component,1,0)
        sitk_write_lab(fix_label_array,fixed_lab,out_dir,get_name_wo_suffix(fix_label_path))

        warp_mv_label_array=np.where(sitk.GetArrayFromImage(warp_mv_label)==self.args.component,1,0)
        sitk_write_lab(warp_mv_label_array,warp_mv_label,out_dir,get_name_wo_suffix(move_label_path))

        ds=calculate_binary_dice(fix_label_array,warp_mv_label_array)
        hd=calculate_binary_hd(fix_label_array,warp_mv_label_array,spacing=fixed_lab.GetSpacing())
        return ds,hd


    def validate(self):
        dirs=sort_glob(self.args.sample_dir+"/*")
        DS=[]
        HD=[]
        for d in dirs:
            target_file=sort_glob(d+"/*%s*label*"%self.args.Ttarget)
            atlas_file=sort_glob(d+"/*%s*label*"%self.args.Tatlas)
            fix_label=sitk.ReadImage(target_file[0])
            fix_array=sitk.GetArrayFromImage(fix_label)
            for itr in atlas_file:
                mv_img=sitk.ReadImage(itr)
                mv_array=sitk.GetArrayFromImage(mv_img)
                ds=calculate_binary_dice(fix_array,mv_array)
                hd=calculate_binary_hd(fix_array,mv_array,spacing=fix_label.GetSpacing())
                print(ds)
                DS.append(ds)
                HD.append(hd)
        outpu2excel(self.args.res_excel, self.args.MOLD_ID + "_DS", DS)
        outpu2excel(self.args.res_excel, self.args.MOLD_ID + "_HD", HD)



