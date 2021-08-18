import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


# class NMI:
#
#     def __init__(self, bin_centers, vol_size, sigma_ratio=0.5, max_clip=1, local=False, crop_background=False, patch_size=1):
#         """
#         Mutual information loss for image-image pairs.
#         Author: Courtney Guo
#
#         If you use this loss function, please cite the following:
#
#         Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
#
#         Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
#         Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
#         MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
#         """
#         print("vxm info: mutual information loss is experimental", file=sys.stderr)
#         self.vol_size = vol_size
#         self.max_clip = max_clip
#         self.patch_size = patch_size
#         self.crop_background = crop_background
#         self.mi = self.local_mi if local else self.global_mi
#         self.vol_bin_centers = (bin_centers)
#         self.num_bins = len(bin_centers)
#         self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
#         self.preterm = (1 / (2 * np.square(self.sigma)))
#
#     def local_mi(self, y_true, y_pred):
#         # reshape bin centers to be (1, 1, B)
#         o = [1, 1, 1, 1, self.num_bins]
#         vbc = K.reshape(self.vol_bin_centers, o)
#
#         # compute padding sizes
#         patch_size = self.patch_size
#         x, y, z = self.vol_size
#         x_r = -x % patch_size
#         y_r = -y % patch_size
#         z_r = -z % patch_size
#         pad_dims = [[0,0]]
#         pad_dims.append([x_r//2, x_r - x_r//2])
#         pad_dims.append([y_r//2, y_r - y_r//2])
#         pad_dims.append([z_r//2, z_r - z_r//2])
#         pad_dims.append([0,0])
#         padding = tf.constant(pad_dims)
#
#         # compute image terms
#         # num channels of y_true and y_pred must be 1
#         I_a = K.exp(- self.preterm * K.square(tf.pad(y_true, padding, 'CONSTANT')  - vbc))
#         I_a /= K.sum(I_a, -1, keepdims=True)
#
#         I_b = K.exp(- self.preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT')  - vbc))
#         I_b /= K.sum(I_b, -1, keepdims=True)
#
#         I_a_patch = tf.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
#         I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
#         I_a_patch = tf.reshape(I_a_patch, [-1, patch_size**3, self.num_bins])
#
#         I_b_patch = tf.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
#         I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
#         I_b_patch = tf.reshape(I_b_patch, [-1, patch_size**3, self.num_bins])
#
#         # compute probabilities
#         I_a_permute = K.permute_dimensions(I_a_patch, (0,2,1))
#         pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
#         pab /= patch_size**3
#         pa = tf.reduce_mean(I_a_patch, 1, keepdims=True)
#         pb = tf.reduce_mean(I_b_patch, 1, keepdims=True)
#
#         papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
#         return K.mean(K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1))
#
#     def global_mi(self, y_true, y_pred):
#         if self.crop_background:
#             # does not support variable batch size
#             thresh = 0.0001
#             padding_size = 20
#             filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])
#
#             smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
#             mask = smooth > thresh
#             # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
#             y_pred = tf.boolean_mask(y_pred, mask)
#             y_true = tf.boolean_mask(y_true, mask)
#             y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
#             y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)
#
#         else:
#             # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
#             y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
#             y_true = K.expand_dims(y_true, 2)
#             y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
#             y_pred = K.expand_dims(y_pred, 2)
#
#         nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)
#
#         # reshape bin centers to be (1, 1, B)
#         o = [1, 1, np.prod(self.vol_bin_centers.get_shape().as_list())]
#         vbc = K.reshape(self.vol_bin_centers, o)
#
#         # compute image terms
#         I_a = K.exp(- self.preterm * K.square(y_true  - vbc))
#         I_a /= K.sum(I_a, -1, keepdims=True)
#
#         I_b = K.exp(- self.preterm * K.square(y_pred  - vbc))
#         I_b /= K.sum(I_b, -1, keepdims=True)
#
#         # compute probabilities
#         I_a_permute = K.permute_dimensions(I_a, (0,2,1))
#         pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
#         pab /= nb_voxels
#         pa = tf.reduce_mean(I_a, 1, keepdims=True)
#         pb = tf.reduce_mean(I_b, 1, keepdims=True)
#
#         papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
#         return K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)
#
#     def loss(self, y_true, y_pred):
#         y_pred = K.clip(y_pred, 0, self.max_clip)
#         y_true = K.clip(y_true, 0, self.max_clip)
#         return -self.mi(y_true, y_pred)