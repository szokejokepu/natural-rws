from core.networks.double_nodes.AbstractDoubleNode import AbstractDoubleNode
from core.networks.nodes.AveragePoolingInverseNode import AveragePoolingInverseNode
from core.networks.nodes.AveragePoolingNode import AveragePoolingNode


class AveragePoolingDoubleNode(AbstractDoubleNode):

    def __init__(self, bottom_layer_shape, pool_size, strides, padding, inverse_type, b_size,
                 n_z_samples,
                 b_size_int, n_z_samples_int, clip_probs=0, pm=True, initializers={}, regularizers={},
                 automatic_diff=False):
        super().__init__(b_size=b_size, n_z_samples=n_z_samples, b_size_int=b_size_int, n_z_samples_int=n_z_samples_int,
                         clip_probs=clip_probs, pm=pm,
                         initializers=initializers, regularizers=regularizers,
                         automatic_diff=automatic_diff)

        self._bottom_layer_shape = bottom_layer_shape

        self._filters_in = self._bottom_layer_shape[-1]
        self._filters_out = self._bottom_layer_shape[-1]
        self._pool_size = pool_size
        self._padding = padding
        self._inverse_type = inverse_type
        self._strides = strides if strides is not None else self._pool_size

        self.ap_rec = AveragePoolingNode(output_size=self._filters_in, pool_size=self._pool_size,
                                         padding=self._padding, strides=self._strides,
                                         input_shape=self._bottom_layer_shape,
                                         clip_probs=self._clip_probs,
                                         pm=self._pm)

        self.ap_gen = AveragePoolingInverseNode(filter_size=self._filters_out, pool_size=self._pool_size,
                                                padding=self._padding, strides=self._strides,
                                                input_shape=self._bottom_layer_shape, inverse_type=self._inverse_type,
                                                clip_probs=self._clip_probs,
                                                pm=self._pm)

    @property
    def rec_distr(self):
        return self.ap_rec

    @property
    def gen_distr(self):
        return self.ap_gen

    def _postprocess_output(self, input):
        return input

    def _preprocess_input(self, input):
        return input

    def get_output_shape(self):
        # [(W−K+2P)/S]+1.
        #
        # W is the input volume - in your case 128
        # K is the Kernel size - in your case 5
        # P is the padding - in your case 0 i believe
        # S is the stride - which you have not provided.

        # TODO this will be a problem when we'll have not only a padding of one
        pad = 1 if self._padding == "same" else 0
        new_rows = (self._bottom_layer_shape[0] - self._pool_size[0] + 2 * pad) // self._strides[0] + 1
        new_cols = (self._bottom_layer_shape[1] - self._pool_size[1] + 2 * pad) // self._strides[1] + 1
        return (new_rows, new_cols, self._filters_in)
        # return self.rec_distr.output_shape()[1:]
