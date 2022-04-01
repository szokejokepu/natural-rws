import numpy as np
import tensorflow as tf

from core.networks.double_nodes.AbstractNaturalDifferentiableDoubleNode import AbstractNaturalDifferentiableDoubleNode
from core.networks.nodes.BernoulliDenseNode import BernoulliDenseNode


class DenseDoubleNode(AbstractNaturalDifferentiableDoubleNode):

    def __init__(self, size_bottom, size_top, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._size_bottom = np.prod(size_bottom)
        self._size_top = size_top

        self.d_rec = BernoulliDenseNode(self._size_top, **kwargs)

        self.d_gen = BernoulliDenseNode(self._size_bottom, **kwargs)

    @property
    def rec_distr(self):
        return self.d_rec

    @property
    def gen_distr(self):
        return self.d_gen

    # Reshape to original shape
    def _postprocess_output(self, input):
        if len(input.shape) != len(self._input_orig_shape) + 1:
            return tf.reshape(input, [-1] + self._input_orig_shape)
        return input

    # Flatten for input to dense
    def _preprocess_input(self, input):
        if len(input.shape) > 2:
            input_flat = tf.reshape(input, [-1, self._input_shape])
            return input_flat
        return input

    def get_output_shape(self):
        return (self._size_top,)

    def manual_diff(self, inputs, probs_of_input, based_on_samples=None, weights=None):
        if weights is None:
            weights = tf.ones(self.mega_batch_size)
        else:
            weights = weights * tf.cast(self.n_z_samples, dtype=tf.float32)

        if self._pm:
            inputs = (inputs + 1) / 2

        difference_b = inputs - probs_of_input

        difference_w = tf.einsum("bu,bv->bvu", difference_b, based_on_samples)
        return self._reduce_grads_with_weights(weights, difference_b), \
               self._reduce_grads_with_weights(weights, difference_w)

    def _reduce_grads_with_weights(self, weights, grads):
        if len(grads.shape.as_list()) == 3:
            return tf.reduce_mean(tf.reduce_mean(tf.reshape(tf.einsum("b,buv->buv", weights, grads),
                                                            [self.n_z_samples, self.b_size,
                                                             *grads.shape.as_list()[1:]]), axis=0), axis=0)
        elif len(grads.shape.as_list()) == 2:
            return tf.reduce_mean(tf.reduce_mean(tf.reshape(tf.einsum("b,bv->bv", weights, grads),
                                                            [self.n_z_samples, self.b_size,
                                                             grads.shape.as_list()[1]]), axis=0), axis=0)
        elif len(grads.shape.as_list()) == 4:
            return tf.reduce_mean(tf.reduce_mean(tf.reshape(tf.einsum("b,buvm->buvm", weights, grads),
                                                            [self.n_z_samples, self.b_size,
                                                             *grads.shape.as_list()[1:]]), axis=0), axis=0)
        elif len(grads.shape.as_list()) == 5:
            return tf.reduce_mean(tf.reduce_mean(tf.reshape(tf.einsum("b,buvmn->buvmn", weights, grads),
                                                            [self.n_z_samples, self.b_size,
                                                             grads.shape.as_list()[1]]), axis=0), axis=0)
        else:
            raise Exception("No such format for variables")
