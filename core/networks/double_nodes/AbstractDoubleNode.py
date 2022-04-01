from abc import abstractmethod

import numpy as np
import sonnet as snt

from core.networks.nodes.AbstractHMNode import AbstractHMNode


class AbstractDoubleNode(AbstractHMNode):

    def __init__(self, b_size, n_z_samples, b_size_int, n_z_samples_int, clip_probs=0, pm=True, initializers={},
                 regularizers={},
                 automatic_diff=False):
        super().__init__(b_size=b_size, n_z_samples=n_z_samples, b_size_int=b_size_int, n_z_samples_int=n_z_samples_int,
                         clip_probs=clip_probs, pm=pm,
                         initializers=initializers, regularizers=regularizers, automatic_diff=automatic_diff)

    @property
    @abstractmethod
    def rec_distr(self):
        pass

    @property
    @abstractmethod
    def gen_distr(self):
        pass

    def _build(self, recognition_input, next_node):
        self._input_orig_shape = recognition_input.shape.as_list()[1:]
        self._input_shape = np.prod(self._input_orig_shape)

        preprocessed_recognition_input = self._preprocess_input(recognition_input)

        recognition_output = self._wake(preprocessed_recognition_input)

        preprocessed_recognition_output = self._preprocess_output(recognition_output)

        generation_input = next_node(preprocessed_recognition_output)

        postprocessed_generation_input = self._postprocess_input(generation_input)

        generation_ouput = self._sleep(postprocessed_generation_input)

        return self._postprocess_output(generation_ouput)

    @snt.reuse_variables
    def _wake(self, inputs):
        self.inputs_r = inputs

        print("recognition_input_flat", inputs.shape)
        self.sample_w_q, self.mean_w_q, self.probs_w_q = self.rec_distr(self.inputs_r)
        print("recognition_output", self.sample_w_q.shape)

        self.sample_w_p, self.mean_w_p, self.probs_w_p = self.gen_distr(self.sample_w_q)

        self.loss_w = self.gen_distr.get_loss(self.inputs_r)
        self.loss_w_s = self.rec_distr.get_loss(self.sample_w_q)

        self.variables_w = self.gen_distr.trainable_variables
        return self.sample_w_q

    @snt.reuse_variables
    def _sleep(self, inputs):
        self.inputs_p = inputs
        print("generation_input", inputs.shape)
        self.sample_s_p, self.mean_s_p, self.probs_s_p = self.gen_distr(self.inputs_p)
        print("generation_ouput", self.sample_s_p.shape)

        self.sample_s_q, self.mean_s_q, self.probs_s_q = self.rec_distr(self.sample_s_p)
        self.loss_s = self.rec_distr.get_loss(self.inputs_p)

        self.variables_s = self.rec_distr.trainable_variables
        return self.sample_s_p

    def wake(self, **kwargs):
        return ([], [])

    def wake_phase_sleep(self, **kwargs):
        return ([], [])

    def sleep(self, **kwargs):
        return ([], [])

    def gen(self):
        return self.sample_s_p, self.mean_s_p

    def rec(self):
        return self.sample_w_q, self.mean_w_q

    def log_prob_gen(self, inputs, based_on):
        self.gen_distr(based_on)
        output = self.gen_distr.log_probs(self._preprocess_input(inputs))
        return output

    def log_prob_rec(self, inputs, based_on):
        self.rec_distr(based_on)
        output = self.rec_distr.log_probs(self._preprocess_input(inputs))
        return output

    def prob_gen(self, inputs):
        output = self.gen_distr.get_probs(inputs)
        return output

    def prob_rec(self, inputs):
        output = self.rec_distr.get_probs(self._preprocess_input(inputs))
        return output

    def shape(self):
        return (self.rec_distr.shape(), self.gen_distr.shape())

    def _postprocess_output(self, input):
        return input

    def _preprocess_input(self, input):
        return input

    def _postprocess_input(self, input):
        return input

    def _preprocess_output(self, input):
        return input

    @abstractmethod
    def get_output_shape(self):
        pass

    def manual_diff(self, inputs, probs_of_input, based_on_samples=None, weights=None):
        return []

    def auto_diff(self, loss, variables, stops, weights=None):
        return []
