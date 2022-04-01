import numpy as np
import sonnet as snt
import tensorflow as tf

from core.networks.nodes.AbstractHMNode import D_TYPE
from core.networks.double_nodes.AbstractDifferentiableDoubleNode import AbstractDifferentiableDoubleNode
from core.networks.nodes.BernoulliDenseNode import BernoulliDenseNode


class BiasNode(AbstractDifferentiableDoubleNode):
    def __init__(self, size_bottom, **kwargs):
        super(BiasNode, self).__init__(**kwargs)

        self._size_bottom = size_bottom

        self.bias = tf.constant([[0.0]])
        # self.bias = tf.constant([[0.0]]) if self._pm else tf.constant([[0.5]])

        new_dict = kwargs
        new_dict["initializers"] = {
            "w": tf.constant_initializer(0.0),
            "b": self._initializers["b"]}

        # self.d_gen = BernoulliDenseNode(self._size_bottom, initializers={
        #     "w": tf.constant_initializer(0.0),
        #     "b": self._initializers["b"]},
        #                                 regularizers=self._regularizers)
        self.d_gen = BernoulliDenseNode(self._size_bottom, **new_dict)

    @property
    def gen_distr(self):
        return self.d_gen

    @property
    def rec_distr(self):
        return None

    def _build(self, recognition_input, **kwargs):
        self._input_orig_shape = recognition_input.shape.as_list()[1:]
        self._input_shape = np.prod(self._input_orig_shape)

        self.size_prior = tf.compat.v1.placeholder_with_default(self.mega_batch_size, shape=None)
        self._tiled_bias = tf.tile(self.bias, [self.size_prior, 1])

        preprocessed_recognition_input = self._preprocess_input(recognition_input)
        self._wake(preprocessed_recognition_input)
        self._preprocessed_recognition_output = self._postprocess_output(self.sample_w_p)
        self._preprocessed_recognition_output_mean = self._postprocess_output(self.mean_w_p)
        return self._preprocessed_recognition_output

    @snt.reuse_variables
    def _wake(self, inputs):
        self.inputs_r = inputs

        self.inputs_p = self._tiled_bias
        self.sample_w_p, self.mean_w_p, self.probs_w_p = self.gen_distr(self.inputs_p)

        self.loss_w = self.gen_distr.get_loss(self.inputs_r)

        self.variables_w = self.gen_distr.trainable_variables[0:1]
        return None

    def wake(self, use_natural=False, weights=None, global_step=None, layer_name="", imp_weights=None,
             fisher_calculator=None):
        if self._automatic_diff:
            self._gradients_w = self.auto_diff(self.loss_w, self.variables_w, stops=self.bias, weights=weights)
        else:
            self._gradients_w = self.manual_diff(self.inputs_r, self.probs_w_p, weights=weights)

        if use_natural:
            self._natural_gradients_w = self._apply_fisher_multipliers(fisher_calculator=fisher_calculator,
                                                               next_layer_distr_probs=self.prob_gen(self.sample_w_p),
                                                               previous_layer_sample=None,
                                                               grads=self._gradients_w,
                                                               global_step=global_step,
                                                               layer=layer_name,
                                                               imp_weights=imp_weights)  # only one that needs this
            return self._natural_gradients_w, self.variables_w
        else:
            return self._gradients_w, self.variables_w

    def _sleep(self, inputs):
        raise Exception("No sleep for the Bias")

    def wake_phase_sleep(self, **kwargs):
        raise Exception("No wake_phase_sleep for the Bias")

    def sleep(self, **kwargs):
        raise Exception("No sleep for the Bias")

    def gen(self):
        return self._preprocessed_recognition_output, self._preprocessed_recognition_output_mean

    def rec(self):
        raise Exception("Bias cannot be called as recognition node")

    def log_prob_gen(self, inputs, based_on):
        self.gen_distr(based_on)
        output = self.gen_distr.log_probs(self._preprocess_input(inputs))
        return output

    def log_prob_rec(self, inputs, based_on):
        raise Exception("Bias cannot be called as recognition node")

    def prob_gen(self, inputs):
        output = self.gen_distr.get_probs(inputs)
        return output

    def prob_rec(self, inputs):
        raise Exception("Bias cannot be called as recognition node")

    def mean(self):
        return self.gen_distr.mean()

    def sample(self, size=()):
        return self.gen_distr.sample(size)

    def shape(self):
        return (self.gen_distr.shape())

    def create_id(self):
        return "B{}".format(self._size_bottom)

    def manual_diff(self, inputs, probs_of_input, based_on_samples=None, weights=None):
        if weights is None:
            weights = tf.ones(self.mega_batch_size)

        if self._pm:
            inputs = (inputs + 1) / 2

        difference_b = inputs - probs_of_input
        return self._reduce_grads_with_weights(weights, difference_b)

    def _reduce_grads_with_weights(self, weights, grads):
        if len(grads.shape.as_list()) == 2:
            return tf.reduce_mean(tf.reduce_sum(tf.reshape(tf.einsum("b,bv->bv", weights, grads),
                                                           [self.n_z_samples, self.b_size,
                                                            grads.shape.as_list()[1]]), axis=0), axis=0)
        else:
            raise Exception("No such format for variables")

    def _apply_fisher_multipliers(self, fisher_calculator, next_layer_distr_probs, previous_layer_sample, grads,
                                  global_step, layer, imp_weights=None):
        (difference_b,) = grads
        weight_b, weight_w = self.trainable_variables
        # The formula is F=E[-q*(1-q)[h|1]^T[h|1]]
        # assert not np.bitwise_xor(previous_layer_sample is not None,
        #                           difference_w is not None), "In the case of the bias there's no grad_w and previous layer"

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

        weight_b = tf.cast(weight_b, dtype=D_TYPE)
        fisher_weights_reduced = fisher_calculator._alternate_node(
            value_fn=lambda: [tf.reduce_sum(get_fisher_weights(), axis=0)],
            shapes=[[next_layer_distr_probs.shape.as_list()[-1]]],
            var_names=[layer + "B"],
            global_step=global_step,
            dtype=next_layer_distr_probs.dtype)

        alpha = tf.cast(fisher_calculator._diagonal_pad, dtype=D_TYPE)
        if fisher_calculator._d_p is None or fisher_calculator._d_p == 0.0:
            grads_b = difference_b / fisher_weights_reduced
        else:
            grads_b = fisher_calculator._damping_multiplier(
                lambda: difference_b * (1.0 + alpha) / (alpha + fisher_weights_reduced),
                lambda: difference_b)
        grads_b = tf.cast(grads_b, dtype=orig_dtype)
        fisher_calculator._nat_reg.append(
            tf.cast(fisher_calculator._n_reg * weight_b * alpha * weight_b, dtype=orig_dtype)[0])

        return grads_b,

    def get_output_shape(self):
        return ()

    # Flatten for input to dense
    def _preprocess_input(self, input):
        if len(input.shape) > 2:
            input_flat = tf.reshape(input, [-1, self._size_bottom])
            return input_flat
        return input

    # Reshape to original shape
    def _postprocess_output(self, input):
        if len(input.shape) != len(self._input_orig_shape) + 1:
            return tf.reshape(input, [-1] + self._input_orig_shape)
        return input
