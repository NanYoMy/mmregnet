import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from medpy.metric import hd,asd

from config.Defines import Get_Name_By_Index
from dirutil.helper import get_name_wo_suffix
from evaluate.metric import calculate_binary_dice, neg_jac, print_mean_and_std
from excelutil.output2excel import outpu2excel
from tfop import utils as util, layers as layer, losses as loss
from tfop.losses import restore_loss
from learn2reg.challenge_sampler import CHallengeSampler
from learn2reg.loss import NVISimilarity
from learn2reg.sampler import MMSampler
from model.base_model import BaseModelV2
from sitkImageIO.itkdatawriter import sitk_write_lab,sitk_write_images,sitk_write_labs


class MMReg_base(BaseModelV2):
    def __init__(self,sess,args):
        BaseModelV2.__init__(self, sess, args)

        self.train_sampler = MMSampler(self.args, 'train')
        self.validate_sampler = MMSampler(self.args, 'validate')
        self.minibatch_size = self.args.batch_size
        self.image_size = [self.args.image_size, self.args.image_size, self.args.image_size]
        self.grid_ref = util.get_reference_grid(self.image_size)
        if args.phase == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.build_network()
        self.summary()
    def warp_image(self, input_,ddf):
        return util.resample_linear(input_, self.grid_ref+ ddf)

    def _regnet(self, mv_img,mv_lab, fix_img,fix_lab, reuse=False,scop_name="shared_regnet"):
        input_layer = tf.concat([layer.resize_volume(mv_img, self.image_size), fix_img], axis=4)
        ddf_levels = [0, 1, 2, 3, 4]
        self.num_channel_initial = self.args.num_channel_initial
        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        min_level = min(ddf_levels)
        with tf.variable_scope(scop_name,reuse=reuse):
            h0, hc0 = layer.downsample_resnet_block(self.is_train, input_layer, 2, nc[0], k_conv0=[7, 7, 7],name='local_down_0')
            h1, hc1 = layer.downsample_resnet_block(self.is_train, h0, nc[0], nc[1], name='local_down_1')
            h2, hc2 = layer.downsample_resnet_block(self.is_train, h1, nc[1], nc[2], name='local_down_2')
            h3, hc3 = layer.downsample_resnet_block(self.is_train, h2, nc[2], nc[3], name='local_down_3')
            hm = [layer.conv3_block(self.is_train, h3, nc[3], nc[4], name='local_deep_4')]
            hm += [layer.upsample_resnet_block(self.is_train, hm[0], hc3, nc[4], nc[3],name='local_up_3')] if min_level < 4 else []
            hm += [layer.upsample_resnet_block(self.is_train, hm[1], hc2, nc[3], nc[2],name='local_up_2')] if min_level < 3 else []
            hm += [layer.upsample_resnet_block(self.is_train, hm[2], hc1, nc[2], nc[1],name='local_up_1')] if min_level < 2 else []
            hm += [layer.upsample_resnet_block(self.is_train, hm[3], hc0, nc[1], nc[0],name='local_up_0')] if min_level < 1 else []
            ddf_list = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf1_sum_%d' % idx) for idx in ddf_levels]
            ddf_list = tf.stack(ddf_list, axis=5)
            ddf_MV_FIX = tf.reduce_sum(ddf_list, axis=5)

            ddf_list2 = [layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='ddf2_sum_%d' % idx) for idx in ddf_levels]
            ddf_list2 = tf.stack(ddf_list2, axis=5)
            ddf_FIX_MV = tf.reduce_sum(ddf_list2, axis=5)

        w_mv_img = self.warp_image(mv_img, ddf_MV_FIX)
        w_mv_lab = self.warp_image(mv_lab, ddf_MV_FIX)
        r_mv_img = self.warp_image(w_mv_img, ddf_FIX_MV)

        w_fix_img = self.warp_image(fix_img, ddf_FIX_MV)
        w_fix_lab = self.warp_image(fix_lab, ddf_FIX_MV)
        r_fix_img = self.warp_image(w_fix_img, ddf_MV_FIX)
        return ddf_MV_FIX,ddf_FIX_MV,w_mv_img,w_mv_lab,r_mv_img,w_fix_img,w_fix_lab,r_fix_img

    def cal_nvi_loss(self,w_mv_img,i_fix_img,w_fix_img,i_mv_img):
        nvi_loss_1 = self.multiScaleNVILoss(w_mv_img, i_fix_img)
        nvi_loss_2 = self.multiScaleNVILoss(w_fix_img, i_mv_img)
        nvi_loss = nvi_loss_1 + nvi_loss_2
        return nvi_loss

    def consis_loss(self,i_mv_img,r_mv_img,i_fix_img,r_fix_img):
        consistent =  (restore_loss(i_mv_img, r_mv_img) + restore_loss(i_fix_img, r_fix_img))
        return consistent

    def bend_loss(self,ddf_mv_f,ddf_f_mv):
        # create loss
        ddf1_bend = tf.reduce_mean(loss.local_displacement_energy(ddf_mv_f, 'bending', 1))
        ddf2_bend = tf.reduce_mean(loss.local_displacement_energy(ddf_f_mv, 'bending', 1))
        ddf_bend = (ddf1_bend + ddf2_bend)
        return ddf_bend

    def multiScaleNVILoss(self, warped_mv1_img, input_FIX_image):
        grad_loss=0
        scales=[1,2,3]
        for s in scales :
            grad_loss=grad_loss+NVISimilarity(warped_mv1_img, input_FIX_image, s)
        return grad_loss/len(scales)

    def train(self):
        self.is_train=True
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        for glob_step in range(self.args.iteration):
            mv_img1s, mv_lab1s, mv_img2s, mv_lab2s, fix_imgs, fix_labs=self.train_sampler.next_sample()
            trainFeed = self.create_feed_dict(mv_img1s, mv_lab1s, mv_img2s, mv_lab2s, fix_imgs, fix_labs, is_aug=True)

            _,nv_loss,cyc_consis,bend,multi_consis,summary=self.sess.run([self.train_op, self.nvi_loss, self.cycle_consistent, self.ddf_bend, self.multi_consis, self.summary_all], feed_dict=trainFeed)
            self.writer.add_summary(summary,glob_step)
            self.logger.debug("step %d: nv_loss=%f,cyc_consis=%f,bend=%f,multi_consis=%f"%(glob_step,nv_loss,cyc_consis,bend,multi_consis))

            if np.mod(glob_step, self.args.print_freq) == 1:
                # self.sample(glob_step)
                self.validate_set()
            if np.mod(glob_step, self.args.save_freq) == 1:
                self.save(self.args.checkpoint_dir, glob_step)

    def summary(self):
        tf.summary.scalar("nvi_loss_1",self.nvi1)
        tf.summary.scalar("nvi_loss_2", self.nvi2)
        tf.summary.scalar("ddf1_bend", self.bend1)
        tf.summary.scalar("ddf2_bend",self.bend2)
        tf.summary.scalar('multi_consis', self.multi_consis)
        tf.summary.scalar("cycle_consis", self.cycle_consistent)
        # tf.summary.scalar("anti_folding_loss", self.anti_folding_loss)
        tf.summary.image("fix_img", tf.expand_dims(self.i_fix_img[:, :, 48, :, 0], -1))
        tf.summary.image("warped_fix_img", tf.expand_dims(self.w_fix1_img[:, :, 48, :, 0], -1))
        tf.summary.image("mv1_img", tf.expand_dims(self.i_mv1_img[:, :, 48, :, 0], -1))
        tf.summary.image("warped_mv1_img", tf.expand_dims(self.w_mv1_img[:, :, 48, :, 0], -1))
        tf.summary.image("mv2_img", tf.expand_dims(self.i_mv2_img[:, :, 48, :, 0], -1))
        tf.summary.image("warped_mv2_img", tf.expand_dims(self.w_mv2_img[:, :, 48, :, 0], -1))
        self.summary_all=tf.summary.merge_all()

    def sample(self, iter,write_img=False):

        p_img_mv1s,p_lab_mv1s, p_img_mv2s,p_lab_mv2s,p_img_fixs, p_lab_fixs = self.validate_sampler.get_data_path()
        img_mv1s, lab_mv1s, img_mv2s, lab_mv2s, img_fixs, lab_fixs  = self.validate_sampler.get_batch_data(p_img_mv1s,p_lab_mv1s, p_img_mv2s,p_lab_mv2s,p_img_fixs, p_lab_fixs)
        trainFeed = self.create_feed_dict(img_mv1s, lab_mv1s, img_mv2s, lab_mv2s, img_fixs, lab_fixs,is_aug=False)
        warped_mv1_lab,warped_mv2_lab,input_mv_lab1,input_mv_lab2,input_fix_lab=self.sess.run([self.w_mv1_lab, self.w_mv2_lab, self.i_mv1_lab,self.i_mv2_lab, self.i_fix_lab], feed_dict=trainFeed)

        if write_img:


            sitk_write_labs(warped_mv1_lab, None, self.args.sample_dir, '%d_warped_mv1_lab' % (iter))
            sitk_write_labs(warped_mv1_lab, None, self.args.sample_dir, '%d_warped_mv2_lab' % (iter))
            sitk_write_labs(input_fix_lab, None, self.args.sample_dir, '%d_fixe_lab' % (iter))
            warped_mv1_img, warped_mv2_img, input_fix_img,input_mv1_img,input_mv2_img ,i_mv1_lab,i_mv2_lab= self.sess.run([self.w_mv1_img, self.w_mv2_img,  self.i_fix_img,self.i_mv1_img,self.i_mv2_img,self.i_mv1_lab,self.i_mv2_lab], feed_dict=trainFeed)
            sitk_write_images(warped_mv1_img, None, self.args.sample_dir, '%d_warped_mv1_img' % (iter))
            sitk_write_images(input_mv1_img, None, self.args.sample_dir, '%d_input_mv1_img' % (iter))
            sitk_write_labs(i_mv1_lab, None, self.args.sample_dir, '%d_input_mv1_lab' % (iter))
            sitk_write_images(warped_mv2_img, None, self.args.sample_dir, '%d_warped_mv2_img' % (iter))
            sitk_write_images(input_mv2_img, None, self.args.sample_dir, '%d_input_mv2_img' % (iter))
            sitk_write_labs(i_mv2_lab, None, self.args.sample_dir, '%d_input_mv2_lab' % (iter))
            sitk_write_images(input_fix_img, None, self.args.sample_dir, '%d_fixe_img' % (iter))

        dice_before_reg1 = calculate_binary_dice(input_mv_lab1, input_fix_lab)
        dice_before_reg2 = calculate_binary_dice(input_mv_lab2, input_fix_lab)
        warp_mv1_dice=calculate_binary_dice(warped_mv1_lab, input_fix_lab)
        warp_mv2_dice=calculate_binary_dice(warped_mv2_lab, input_fix_lab)

        para=sitk.ReadImage(p_lab_fixs[0])
        mv1_hd=asd(np.squeeze(warped_mv1_lab[0,...]),np.squeeze(input_fix_lab[0,...]),voxelspacing=para.GetSpacing())
        mv2_hd=asd(np.squeeze(warped_mv2_lab[0,...]),np.squeeze(input_mv_lab1[0,...]),voxelspacing=para.GetSpacing())

        ddf_mv1_f,ddf_mv2_f=self.sess.run([self.ddf_mv1_f, self.ddf_f_mv1], feed_dict=trainFeed)
        _,_,neg_ddf_mv1_f=neg_jac(ddf_mv1_f[0,...])
        _,_,neg_ddf_mv2_f=neg_jac(ddf_mv2_f[0,...])
        self.logger.debug("test_step %d: before_reg_dice=%f, mv1_dice =%f , mv2_dice=%f, mv1_hd=%f, mv2_hd=%f neg_jac %d %d"%(iter,dice_before_reg1,warp_mv1_dice,warp_mv2_dice,mv1_hd,mv2_hd,neg_ddf_mv1_f,neg_ddf_mv2_f))
        return dice_before_reg1,dice_before_reg2,warp_mv1_dice,warp_mv2_dice,mv1_hd,mv2_hd,neg_ddf_mv1_f,neg_ddf_mv2_f

    def validate(self):
        self.is_train=False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # res={'mv1_dice':[],'mv1_hd':[],'mv2_dice':[],'mv2_hd':[],'neg_ddf1':[],'neg_ddf2':[]}

        self.validate_set(True)

    def validate_set(self ,write_img=False):
        res={'mv1_dice':[],'mv1_asd':[],'mv2_dice':[],'mv2_asd':[],'bf_reg1':[],'bf_reg2':[]}
        for i in range(self.validate_sampler.nb_pairs):
            _bf_reg1,_bf_reg2,_mv1_dice, _mv2_dice, _mv1_hd, _mv2_hd, _neg_ddf1, _neg_ddf2 = self.sample(i,write_img)
            res["mv1_dice"].append(_mv1_dice)
            res["mv2_dice"].append(_mv2_dice)
            res["mv1_asd"].append(_mv1_hd)
            res["mv2_asd"].append(_mv2_hd)
            res["bf_reg1"].append(_bf_reg1)
            res["bf_reg2"].append(_bf_reg2)
            # res["neg_ddf1"].append(_neg_ddf1)
            # res["neg_ddf2"].append(_neg_ddf2)
        print(Get_Name_By_Index(self.args.component))
        print("=============%s================" % (self.args.mode))
        for itr in ['mv1_dice','mv2_dice','mv1_asd','mv2_asd','bf_reg1','bf_reg2']:
            print(itr)
            outpu2excel(self.args.res_excel, self.args.MODEL_ID + "_" + itr, res[itr])
            print_mean_and_std(res[itr], itr)

    def test(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        csample = CHallengeSampler(self.args,self.is_train)
        for atlas_ind in range(csample.len_mv):
            for tgt_ind in range(csample.len_fix):
                fix_imgs, fix_labs,mv_imgs,mv_labs=csample.get_batch_data([atlas_ind],[tgt_ind])
                trainFeed = self.create_feed_dict(fix_imgs, fix_labs, mv_imgs, mv_labs,is_aug=False)
                warp_mv_img, warp_mv_label = self.sess.run([self.w_mv1_img, self.warped_MV_label], feed_dict=trainFeed)
                p_ata=csample.img_mv[atlas_ind]
                p_tgt=csample.img_fix[tgt_ind]

                outputdir= self.args.test_dir+"/atlas_%s/"%(get_name_wo_suffix(p_ata))
                name=get_name_wo_suffix(p_tgt).replace('image','label')

                sitk_write_lab(warp_mv_label[0,...],sitk.ReadImage(p_tgt),outputdir,name)

    def create_feed_dict(self, mv_img1s, mv_lab1s, mv_img2s, mv_lab2s, fix_imgs, fix_labs, is_aug=False):
        trainFeed = {self.ph_mv_img1: mv_img1s,
                     self.ph_mv_lab1: mv_lab1s,
                     self.ph_mv_img2: mv_img2s,
                     self.ph_mv_lab2: mv_lab2s,
                     self.ph_fix_img: fix_imgs,
                     self.ph_fix_lab: fix_labs,
                     self.ph_fixed_affine: util.random_transform_generator(self.args.batch_size),
                     self.ph_moving_affine1: util.random_transform_generator(self.args.batch_size, 0.1),
                     self.ph_moving_affine2: util.random_transform_generator(self.args.batch_size, 0.1),
                     }
        if is_aug==True:
            pass
        else:
            trainFeed = {self.ph_mv_img1: mv_img1s,
                         self.ph_mv_lab1: mv_lab1s,
                         self.ph_mv_img2: mv_img2s,
                         self.ph_mv_lab2: mv_lab2s,
                         self.ph_fix_img: fix_imgs,
                         self.ph_fix_lab: fix_labs,
                         self.ph_fixed_affine: util.initial_transform_generator(self.args.batch_size),
                         self.ph_moving_affine1: util.initial_transform_generator(self.args.batch_size),
                         self.ph_moving_affine2: util.initial_transform_generator(self.args.batch_size),
                         }

        return trainFeed


class MMReg(MMReg_base):
    def build_network(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.args.lr, self.global_step, self.args.decay_freq, 0.96,staircase=True)

        # input
        self.ph_mv_img1 = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_mv_lab1 = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_mv_img2 = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_mv_lab2 = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_fix_img = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])
        self.ph_fix_lab = tf.placeholder(tf.float32, [self.args.batch_size] + self.image_size + [1])

        self.ph_moving_affine1 = tf.placeholder(tf.float32, [self.args.batch_size] + [1, 12])  # 数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
        self.ph_moving_affine2 = tf.placeholder(tf.float32, [self.args.batch_size] + [1, 12])  # 数据进行augment,4x4矩阵，但是最后四个参数为0001，所以一共12个参数
        self.ph_fixed_affine = tf.placeholder(tf.float32, [self.args.batch_size] + [1,12])

        #data augmentation
        self.i_mv1_img, self.i_mv1_lab=util.augment_3Ddata_by_affine(self.ph_mv_img1, self.ph_mv_lab1, self.ph_moving_affine1)
        self.i_mv2_img, self.i_mv2_lab=util.augment_3Ddata_by_affine(self.ph_mv_img2, self.ph_mv_lab2, self.ph_moving_affine2)
        self.i_fix_img, self.i_fix_lab=util.augment_3Ddata_by_affine(self.ph_fix_img, self.ph_fix_lab, self.ph_fixed_affine)

        self.ddf_mv1_f, self.ddf_f_mv1, self.w_mv1_img,self.w_mv1_lab ,self.r_mv1_img, self.w_fix1_img,self.w_fix1_lab, self.r_fix1_img=self._regnet(self.i_mv1_img,self.i_mv1_lab,self.i_fix_img,self.i_fix_lab,scop_name="regA")
        self.ddf_mv2_f, self.ddf_f_mv2, self.w_mv2_img,self.w_mv2_lab ,self.r_mv2_img, self.w_fix2_img,self.w_fix2_lab, self.r_fix2_img=self._regnet(self.i_mv2_img,self.i_mv2_lab,self.i_fix_img,self.i_fix_lab,scop_name='reg_b')
        self.bend1=self.bend_loss(self.ddf_f_mv1,self.ddf_mv1_f)
        self.bend2=self.bend_loss(self.ddf_f_mv2,self.ddf_mv2_f)
        self.ddf_bend=self.bend1+self.bend2

        self.cyc_consis1=self.consis_loss(self.i_mv1_img, self.r_mv1_img, self.i_fix_img, self.r_fix1_img)
        self.cyc_consis2=self.consis_loss(self.i_mv2_img, self.r_mv2_img, self.i_fix_img, self.r_fix2_img)
        self.cycle_consistent = self.cyc_consis1 + self.cyc_consis2
        '''
        #这个和后面的nvil+nvi2重复，因为nvi1会让w_mv1_img和i_fix_img相同，而nvi2会让w_mv2_img和i_fix_img相同.
        等效于w_mv1_img==w_mv2_img
        '''
        # self.consis=restore_loss(self.w_mv1_img, self.w_mv2_img)
        # self.multi_consis=tf.reduce_mean(loss.multi_scale_loss(self.w_mv1_lab, self.w_mv2_lab, 'dice', [0, 1, 2, 4]))
        _warp_mv1_mv2=self.warp_image(self.w_mv1_img,self.ddf_f_mv2)
        _warp_mv2_mv1=self.warp_image(self.w_mv2_img,self.ddf_f_mv1)
        self.multi_consis = self.cal_nvi_loss(self.i_mv1_img, _warp_mv2_mv1, self.i_mv2_img, _warp_mv1_mv2)

        self.nvi1=self.cal_nvi_loss(self.w_mv1_img, self.i_fix_img, self.w_fix1_img, self.i_mv1_img)
        self.nvi2=self.cal_nvi_loss(self.w_mv2_img, self.i_fix_img, self.w_fix2_img, self.i_mv2_img)
        self.nvi_loss= self.nvi1 + self.nvi2

        # self.anti_folding_loss = self.args.lambda_anti* (loss.anti_folding(self.ddf_mv1_f) + loss.anti_folding(self.ddf_f_mv1))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.nvi_loss
            + self.args.lambda_bend*self.ddf_bend
            +self.args.lambda_cycle_consis*self.cycle_consistent
            +self.args.lambda_multi_consis*self.multi_consis,
            global_step=self.global_step)
        self.logger.debug("build network finish")

