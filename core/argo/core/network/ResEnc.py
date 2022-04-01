import tensorflow as tf
import sonnet as snt
from .build_utils import residual_stack, maybe_set_l2_conv_contractive_regularizer
from .AbstractResNetLayer import AbstractResNetLayer

class ResEnc(AbstractResNetLayer):
    """
    res enc used in VQ
    """
#TODO remove biases before batch norm, see if it makes any difference. Remove dropouts?
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 activation,
                 is_training,
                 name='ResEnc',
                 prob_drop=0.1,
                 bn_momentum=0.99,
                 bn_renormalization=True,
                 creg_scale=None,
                 **extra_params):

        super().__init__(num_hiddens,
                         num_residual_layers,
                         num_residual_hiddens,
                         activation,
                         is_training,
                         name=name,
                         prob_drop=prob_drop,
                         bn_momentum=bn_momentum,
                         bn_renormalization=bn_renormalization,
                         creg_scale=creg_scale,
                         **extra_params)

    def _build(self, x):

        # h_pre = x

        conv1 = snt.Conv2D(
                            output_channels=self._num_hiddens / 2,
                            kernel_shape=(4, 4),
                            stride=(2, 2),
                            # use_bias=False,
                            **self._extra_params,
                            name="enc_1")

        h = conv1(x)
        maybe_set_l2_conv_contractive_regularizer(conv1, h, self._activation, self._creg_scale, name="enc_1_creg")

        h = self._dropout(h, training=self._is_training)
        h = tf.layers.batch_normalization(h, training=self._is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_1")

        h = self._activation(h)

        conv2 = snt.Conv2D(
                            output_channels=self._num_hiddens,
                            kernel_shape=(4, 4),
                            stride=(2, 2),
                            # use_bias=False,
                            **self._extra_params,
                            name="enc_2")
        h = conv2(h)
        maybe_set_l2_conv_contractive_regularizer(conv2, h, self._activation, self._creg_scale, name="enc_2_creg")

        h = self._dropout(h, training=self._is_training)
        h = tf.layers.batch_normalization(h, training=self._is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_2")

        h = self._activation(h)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens,
            activation=self._activation,
            training=self._is_training,
            prob_drop=self._prob_drop,
            momentum=self._bn_momentum,
            renorm=self._bn_renormalization,
            renorm_momentum=self._bn_momentum,
            renorm_clipping=self._renorm_clipping,
            creg_scale = self._creg_scale,
            **self._extra_params
        )

        return h


