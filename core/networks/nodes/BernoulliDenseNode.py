import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from core.networks.nodes.AbstractHMNode import D_TYPE
from core.networks.nodes.AbstractNaturalizableStochasticHMNode import AbstractNaturalizableStochasticHMNode
from core.networks.nodes.BernoulliPassthrough import BernoulliPassthrough


class BernoulliDenseNode(AbstractNaturalizableStochasticHMNode):
    def __init__(self, size, **kwargs):
        super(BernoulliDenseNode, self).__init__(**kwargs)
        self._size = size

        self.d = Dense(size,
                       kernel_initializer=self._initializers["w"],
                       kernel_regularizer=(self._regularizers["w"] if "w" in self._regularizers else None),
                       bias_initializer=self._initializers["b"],
                       bias_regularizer=(self._regularizers["b"] if "b" in self._regularizers else None))

        self.bp = BernoulliPassthrough(self._pm, clip_probs=self._clip_probs)

    @property
    def distr(self):
        return self.bp

    @snt.reuse_variables
    def _build(self, inputs, **kwargs):
        logits = self.d(inputs)

        # with tf.control_dependencies([tf.print("BD_input_shape", tf.shape(inputs))]):
        self.distr(logits)

        # with tf.control_dependencies([tf.print("BD_sample_shape", tf.shape(self.distr.sample()))]):
        output = self.distr.sample()
        return output, self.mean(), self.distr.probs

    def _apply_fisher_multipliers(self, fisher_calculator, next_layer_distr_probs, previous_layer_sample, grads,
                                  global_step, layer, imp_weights=None):
        difference_b, difference_w = grads
        weight_b, weight_w = self.trainable_variables
        # The formula is F=E[-q*(1-q)[h|1]^T[h|1]]
        assert not np.bitwise_xor(previous_layer_sample is not None,
                                  difference_w is not None), "In the case of the bias there's no grad_w and previous layer"

        orig_dtype = next_layer_distr_probs.dtype
        next_layer_distr_probs = tf.cast(next_layer_distr_probs, dtype=D_TYPE)
        difference_b = tf.cast(difference_b, dtype=D_TYPE)

        def get_fisher_weights():
            weights = (next_layer_distr_probs * (1 - next_layer_distr_probs)) / tf.cast(self.mega_batch_size,
                                                                                        dtype=D_TYPE)

            if len(weights.shape) == 1:
                weights = tf.expand_dims(weights, axis=0)

            if imp_weights is not None:
                weights = tf.einsum("b,bu->bu", tf.cast(imp_weights, dtype=D_TYPE), weights)

            return weights

        difference_w = tf.cast(difference_w, dtype=D_TYPE)
        bias_pad = tf.constant([[0, 0], [0, 1]])

        previous_layer_sample_transposed = tf.transpose(
            tf.pad(previous_layer_sample, bias_pad, "CONSTANT", constant_values=1), perm=[1, 0])

        diff_concat_transposed = tf.transpose(tf.concat([difference_w, tf.reshape(difference_b, [1, -1])], axis=0),
                                              perm=[1, 0])
        # Only used for natural regression
        weight_concat = tf.concat([weight_w, tf.reshape(weight_b, [1, -1])], axis=0)

        k_len = self.mega_batch_size_int

        grads_w_b_concat = fisher_calculator._multiply_grads_by_fisher_inv(weight_concat=weight_concat,
                                                                           difference=diff_concat_transposed,
                                                                           weights=tf.transpose(get_fisher_weights(),
                                                                                                perm=[1, 0]),
                                                                           previous_layer=previous_layer_sample_transposed,
                                                                           global_step=global_step,
                                                                           orig_dtype=orig_dtype,
                                                                           layer=layer, k_len=k_len)

        assert grads_w_b_concat.shape == diff_concat_transposed.shape, "Shapes of gradients pre and post multiplication are not equal"
        grads_w_b_concat = tf.transpose(grads_w_b_concat, perm=[1, 0])

        grads_w, grads_b = tf.split(grads_w_b_concat, [difference_w.shape.as_list()[0], 1], 0)

        grads_b = tf.reshape(grads_b, [-1])

        assert grads_b.shape == difference_b.shape, "Shapes of gradients pre and post multiplication are not equal for b"
        assert grads_w.shape == difference_w.shape, "Shapes of gradients pre and post multiplication are not equal for W"

        return grads_b, grads_w
