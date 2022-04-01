import tensorflow as tf

from core.networks.double_nodes.AbstractDoubleNode import AbstractDoubleNode


class AbstractDifferentiableDoubleNode(AbstractDoubleNode):

    def __init__(self, b_size, n_z_samples, b_size_int, n_z_samples_int, clip_probs=0, pm=True, initializers={},
                 regularizers={},
                 automatic_diff=False):
        super().__init__(b_size=b_size, n_z_samples=n_z_samples, b_size_int=b_size_int, n_z_samples_int=n_z_samples_int,
                         clip_probs=clip_probs, pm=pm,
                         initializers=initializers, regularizers=regularizers, automatic_diff=automatic_diff)

    def auto_diff(self, loss, variables, stops, weights=None):
        if weights is None:
            weights = tf.ones(tf.shape(loss)) / tf.cast(self.mega_batch_size, dtype=tf.float32)
        else:
            weights = weights / tf.cast(self.b_size, dtype=tf.float32)
        # loss = weights * loss

        # loss = -tf.reduce_mean(loss, axis=0)

        if (len(variables) == 1):
            grads = tf.gradients(-loss, variables, stop_gradients=stops, grad_ys=weights)
            return grads
        else:
            grads = tf.gradients(-loss, variables, stop_gradients=stops, grad_ys=weights)
            return grads

    def manual_diff(self, inputs, probs_of_input, based_on_samples=None, weights=None):
        raise Exception("Not implemented yet, has to be done per layer")

    def wake(self, use_natural=False, weights=None, global_step=None, layer_name="", imp_weights=None,
             fisher_calculator=None):
        if self._automatic_diff:
            self._gradients_w = self.auto_diff(loss=self.loss_w, variables=self.variables_w,
                                                            stops=[self.sample_w_q, self.inputs_r],
                                                            weights=weights)
        else:
            self._gradients_w = self.manual_diff(self.inputs_r, self.probs_w_p, self.sample_w_q,
                                                              weights=weights)

        return self._gradients_w, self.variables_w

    def wake_phase_sleep(self, use_natural=False, weights=None, global_step=None, layer_name="",
                         fisher_calculator=None):

        if self._automatic_diff:
            self._gradients_ws = self.auto_diff(loss=self.loss_w_s, variables=self.variables_s,
                                                              stops=[self.inputs_r],
                                                              weights=weights)  # Loss?
        else:
            self._gradients_ws = self.manual_diff(self.sample_w_q, self.probs_w_q, self.inputs_r,
                                                                weights=weights)
        return self._gradients_ws, self.variables_s

    def sleep(self, use_natural=False, weights=None, global_step=None, layer_name="", fisher_calculator=None):
        if self._automatic_diff:
            self._gradients_s = self.auto_diff(loss=self.loss_s, variables=self.variables_s,
                                                            stops=[self.sample_s_p, self.inputs_p],
                                                            weights=weights)
        else:
            self._gradients_s = self.manual_diff(self.inputs_p, self.probs_s_q, self.sample_s_p,
                                                              weights=weights)
        return self._gradients_s, self.variables_s
