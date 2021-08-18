import numpy as np
import tensorflow as tf

class MutualInformation(object):
    """
    Compute mutual-information-based metrics.
    Inspired by https://github.com/airlab-unibas/airlab/blob/master/airlab/loss/pairwise.py.

    """

    def __init__(self, n_bins=64, sigma=3, **kwargs):
        self.n_bins = n_bins
        self.sigma = 2*sigma**2
        self.kwargs = kwargs
        self.eps = kwargs.pop('eps', 1e-10)
        self.win = kwargs.pop('win', 7); assert self.win % 2 == 1  # window size for local metrics
        self._normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
        self._normalizer_2d = 2.0 * np.pi * sigma ** 2
        self.background_method = kwargs.pop('background_method', 'min')
        if self.background_method is None:
            self.background_value = kwargs.pop('background_value')

    def _compute_marginal_entropy(self, values, bins):
        """
        Compute the marginal entropy using Parzen window estimation.

        :param values: a tensor of shape [n_batch, *vol_shape, channels]
        :param bins: a tensor of shape [n_bins, 1]
        :return: entropy - the marginal entropy;
                 p - the probability distribution
        """
        p = tf.math.exp(-(tf.math.square(tf.reshape(tf.reduce_mean(values, axis=-1), [-1]) - bins) / self.sigma)) / self._normalizer_1d
        p_norm = tf.reduce_mean(p, axis=1)
        p_norm = p_norm / (tf.reduce_sum(p_norm) + self.eps)
        entropy = - tf.reduce_sum(p_norm * tf.math.log(p_norm + self.eps))

        return entropy, p

    def mi(self, target, source):
        """
        Compute mutual information: I(target, source) = H(target) + H(source) - H(target, source).

        :param target:
        :param source:
        :return:
        """
        target=target
        source =source
        if self.background_method == 'min':
            background_fixed = tf.reduce_min(target)
            background_moving = tf.reduce_min(source)
        elif self.background_method == 'mean':
            background_fixed = tf.reduce_mean(target)
            background_moving = tf.reduce_mean(source)
        elif self.background_method is None:
            background_fixed = self.background_value
            background_moving = self.background_value
        else:
            raise NotImplementedError

        bins_target = tf.expand_dims(tf.linspace(background_fixed, tf.reduce_max(target), self.n_bins), axis=-1)
        bins_source = tf.expand_dims(tf.linspace(background_moving, tf.reduce_max(source), self.n_bins), axis=-1)

        # TODO: add masks

        # Compute marginal entropy
        entropy_target, p_t = self._compute_marginal_entropy(target, bins_target)
        entropy_source, p_s = self._compute_marginal_entropy(source, bins_source)

        # compute joint entropy
        p_joint = tf.matmul(p_t, tf.transpose(p_s, perm=[1, 0])) / self._normalizer_2d
        p_joint = p_joint / (tf.reduce_sum(p_joint) + self.eps)

        entropy_joint = - tf.reduce_sum(p_joint * tf.math.log(p_joint + self.eps))

        return -(entropy_target + entropy_source - entropy_joint)
    def nmi(self, target, source):
        """
        Compute normalized mutual information: NMI(target, source) = (H(target) + H(source)) / H(target, source).

        :param target:
        :param source:
        :return:
        """
        target=target
        source =source
        if self.background_method == 'min':
            background_fixed = tf.reduce_min(target)
            background_moving = tf.reduce_min(source)
        elif self.background_method == 'mean':
            background_fixed = tf.reduce_mean(target)
            background_moving = tf.reduce_mean(source)
        elif self.background_method is None:
            background_fixed = self.background_value
            background_moving = self.background_value
        else:
            raise NotImplementedError

        bins_target = tf.expand_dims(tf.linspace(background_fixed, tf.reduce_max(target), self.n_bins), axis=-1)
        bins_source = tf.expand_dims(tf.linspace(background_moving, tf.reduce_max(source), self.n_bins), axis=-1)

        # TODO: add masks

        # Compute marginal entropy
        entropy_target, p_t = self._compute_marginal_entropy(target, bins_target)
        entropy_source, p_s = self._compute_marginal_entropy(source, bins_source)

        # compute joint entropy
        p_joint = tf.matmul(p_t, tf.transpose(p_s, perm=[1, 0])) / self._normalizer_2d
        p_joint = p_joint / (tf.reduce_sum(p_joint) + self.eps)

        entropy_joint = - tf.reduce_sum(p_joint * tf.math.log(p_joint + self.eps))

        return -(entropy_target + entropy_source) / (entropy_joint + self.eps)
    
    def _normalize(self, data):
        data -= tf.reduce_min(data)
        data /= (tf.reduce_max(data) + self.eps)
        return data
    def ecc(self, target, source):
        """
        Compute entropy correlation coefficient: ECC(target, source) = 2 - 2 / NMI(target, source).

        :param target:
        :param source:
        :return:
        """

        return 2 - 2 / (self.nmi(target, source) + self.eps)

    def ce(self, target, source):
        """
        Compute conditional entropy: H(target|source) = H(target, source) - H(source).

        :param target:
        :param source:
        :return:
        """
        if self.background_method == 'min':
            background_fixed = tf.reduce_min(target)
            background_moving = tf.reduce_min(source)
        elif self.background_method == 'mean':
            background_fixed = tf.reduce_mean(target)
            background_moving = tf.reduce_mean(source)
        elif self.background_method is None:
            background_fixed = self.background_value
            background_moving = self.background_value
        else:
            raise NotImplementedError

        bins_target = tf.expand_dims(tf.linspace(background_fixed, tf.reduce_max(target), self.n_bins), axis=-1)
        bins_source = tf.expand_dims(tf.linspace(background_moving, tf.reduce_max(source), self.n_bins), axis=-1)

        # TODO: add masks

        # Compute marginal entropy
        entropy_target, p_t = self._compute_marginal_entropy(target, bins_target)
        entropy_source, p_s = self._compute_marginal_entropy(source, bins_source)

        # compute joint entropy
        p_joint = tf.matmul(p_t, tf.transpose(p_s, perm=[1, 0])) / self._normalizer_2d
        p_joint = p_joint / (tf.reduce_sum(p_joint) + self.eps)

        entropy_joint = - tf.reduce_sum(p_joint * tf.math.log(p_joint + self.eps))

        return entropy_joint - entropy_source