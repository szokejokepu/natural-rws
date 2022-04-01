import tensorflow as tf
import sonnet as snt
from ..utils.argo_utils import get_ac_collection_name
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
import numpy as np

import pdb

kwargs_jacobian = {
    'use_pfor' : False,
    'parallel_iterations': None,
}


def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens, activation,
                   training, prob_drop, momentum, renorm, renorm_momentum, renorm_clipping,
                   initializers, regularizers, creg_scale=None):

    conv = snt.Conv2D(
        output_channels=num_hiddens,
        kernel_shape=(3, 3),
        stride=(1, 1),
        initializers=initializers,
        regularizers=regularizers,
        name="pre")

    h = conv(h)
    maybe_set_l2_conv_contractive_regularizer(conv, h, activation, creg_scale, name="pre_res_creg")

    for i in range(num_residual_layers):

        h_i = activation(h)
        name_i = "res3x3_%d" % i

        conv = snt.Conv2D(
            output_channels=num_residual_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            initializers=initializers,
            regularizers=regularizers,
            name=name_i)

        h_i = conv(h_i)
        maybe_set_l2_conv_contractive_regularizer(conv, h_i, activation, creg_scale, name=name_i+"_creg")

        h_i = tf.layers.batch_normalization(h_i, training=training,
                                          momentum=momentum,
                                          renorm=renorm,
                                          renorm_momentum=renorm_momentum,
                                          renorm_clipping=renorm_clipping,
                                          name="bn_1_%d" % i)
        h_i = tf.layers.dropout(h_i, prob_drop, training=training)

        h_i = activation(h_i)
        name_i = "res1x1_%d" % i

        conv = snt.Conv2D(
                    output_channels=num_hiddens,
                    kernel_shape=(1, 1),
                    stride=(1, 1),
                    initializers=initializers,
                    regularizers=regularizers,
                    name=name_i)

        h_i = conv(h_i)
        maybe_set_l2_conv_contractive_regularizer(conv, h_i, activation, creg_scale, name=name_i+"_creg")

        h_i = tf.layers.batch_normalization(h_i, training=training,
                                          momentum=momentum,
                                          renorm=renorm,
                                          renorm_momentum=renorm_momentum,
                                          renorm_clipping=renorm_clipping,
                                          name="bn_2_%d" % i)
        h_i = tf.layers.dropout(h_i, prob_drop, training=training)

        # h = tf.layers.batch_normalization(h, training=training,
        #                                   momentum=momentum,
        #                                   renorm=renorm,
        #                                   renorm_momentum=renorm_momentum,
        #                                   renorm_clipping=renorm_clipping,
        #                                   name="bn_inc_%d" % i)
        #

        h += h_i

    out = activation(h)
    # maybe_set_l2_contractive_regularizer(out, pre_h, creg_scale)

    return out


def compute_alpha_prime_square(y, activation):
    if not isinstance(activation, str):
        activation = activation.__name__

    if activation == 'relu':
        alpha_prime_square = tf.where_v2(tf.greater(y, 0.), 1., 0.)
    elif activation == 'identity':
        alpha_prime_square = 1.
    elif activation == 'softplus_square':
        alpha_prime_square = tf.square(2 * tf.nn.softplus(y) * tf.nn.sigmoid(y))
    else:
        raise Exception("{:} activation derivative not implemented.")

    return alpha_prime_square


# here
def maybe_set_l2_conv_contractive_regularizer(conv_module, y, activation, scale, g=1., name=None):
    if scale not in [None, 0.]:
        print("computing jacobian for {:}".format(y))

        alpha_prime_square = compute_alpha_prime_square(y, activation)

        gamma = tf.reduce_sum(tf.square(conv_module.w), axis=[0,1,2])

        channels_factor = tf.reduce_sum(g*alpha_prime_square, axis=[1,2]) #tf.stop_gradient()
        
        square_jac = tf.reduce_sum(channels_factor * tf.expand_dims(gamma, axis=0), axis=-1)
        loss = tf.multiply(scale, tf.reduce_mean(square_jac), name=name)
        
        ac_collection_name = get_ac_collection_name()
        tf.compat.v1.add_to_collection(ac_collection_name, loss)

        #pdb.set_trace()
        
def maybe_set_l2_contractive_regularizer_jac(h, x, scale):
    if scale not in [None, 0.]:
        jacobian = batch_jacobian(h, x, **kwargs_jacobian)
        print("computing jacobian {:}".format(jacobian))
        all_axis_but_zero = np.arange(len(jacobian.shape))[1:]
        loss_batch = tf.reduce_sum(tf.square(jacobian), axis=all_axis_but_zero)
        loss = scale*tf.reduce_mean(loss_batch)

        ac_collection_name = get_ac_collection_name()
        tf.compat.v1.add_to_collection(ac_collection_name, loss)
