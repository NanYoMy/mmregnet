import argparse
import tensorflow as tf
from tool.parse import parse_arg_list
tf.set_random_seed(19)
from learn2reg.mmreg import MMReg
from dirutil.helper import mkdir_if_not_exist,mk_or_cleardir
from learn2reg.prepare_mmwhs import prepare_crossvalidation_reg_data,prepare_unsupervised_reg_data
from config.Defines import Get_Name_By_Index
from learn2reg.postprepare_mmwhs import post_process
MOLD_ID='mmwhs'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--iteration', dest='iteration', type=int, default=10001, help='# train iteration')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
parser.add_argument('--image_size', dest='image_size', type=int, default=96, help='the size of image_size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=500, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=500, help='print the debug information every print_freq iterations')

parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')


parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')

parser.add_argument('--lambda_bend', dest='lambda_bend', type=float, default=10, help='defualt 10')
parser.add_argument('--lambda_cycle_consis', dest='lambda_cycle_consis', type=float, default=0.1, help='default 0.1')
parser.add_argument('--lambda_multi_consis', dest='lambda_multi_consis', type=float, default=0.0, help='default 0.1')
parser.add_argument('--component', dest='component', type=int, default=1, help='205=myocardium 500=lv')
parser.add_argument('--task', dest='task', default='multi_chaos', help='MMWHS,multi_chaos')
parser.add_argument('--mode', dest='mode', type=str, default='t1_in_DUAL_mr-t2SPIR_mr-ct', help='atlas1,atlas2,target')

# parser.add_argument('--component', dest='component', type=int, default=205, help='205=myocardium 500=lv')
# parser.add_argument('--task', dest='task', default='MMWHS', help='MMWHS,multi_chaos')
# parser.add_argument('--mode', dest='mode', type=str, default='ct-mr-mr', help='atlas1,atlas2,target')


parser.add_argument('--phase', dest='phase', default='test', help='train,test,validate')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument("--gen_num",dest='gen_num', type=int, nargs=1,default=3000, help="")
parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
args = parser.parse_args()

#label fusion mv

def globel_setup():
    global  args
    DATA_ID = "joint_mmreg_img_consis-%s-%s-%d" % (args.task,args.mode,args.component)
    MODEL_ID = "%s-%f-%f-%f-%d" % (DATA_ID,args.lambda_bend, args.lambda_cycle_consis,args.lambda_multi_consis ,args.fold)
    print("global set %s " % MODEL_ID)
    parser.add_argument('--MODEL_ID', dest='MODEL_ID', default=MODEL_ID,help='path of the dataset')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='../datasets/%s' % (DATA_ID),help='path of the dataset')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='../outputs/%s/checkpoint' % (MODEL_ID),help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='../outputs/%s/sample' % (MODEL_ID),help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='../outputs/%s/test' % (MODEL_ID),help='test sample are saved here')
    parser.add_argument('--log_dir', dest='log_dir', default='../outputs/%s/log' % (MODEL_ID), help='log dir')
    parser.add_argument('--res_excel', dest='res_excel', default='../outputs/result/%s.xls'%(DATA_ID),help='train,test,trainSim,testSim,gen,post')

    args = parser.parse_args()


def main(_):

    globel_setup()
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        prepare_unsupervised_reg_data(args)
        if args.phase == 'train':

            mk_or_cleardir(args.log_dir)
            model =MMReg(sess, args)
            model.train()
        elif args.phase=='validate' or args.phase=='test':
            mk_or_cleardir(args.sample_dir)
            model = MMReg(sess, args)
            model.validate()

        else:
            print("undefined phase")

if __name__ == '__main__':
    # test_code(args)
    tf.app.run()
