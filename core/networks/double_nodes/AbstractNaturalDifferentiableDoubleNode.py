from core.networks.double_nodes.AbstractDifferentiableDoubleNode import AbstractDifferentiableDoubleNode


class AbstractNaturalDifferentiableDoubleNode(AbstractDifferentiableDoubleNode):

    def __init__(self, b_size, n_z_samples, b_size_int, n_z_samples_int, clip_probs=0, pm=True, initializers={},
                 regularizers={},
                 automatic_diff=False):
        super().__init__(b_size=b_size, n_z_samples=n_z_samples, b_size_int=b_size_int, n_z_samples_int=n_z_samples_int,
                         clip_probs=clip_probs, pm=pm,
                         initializers=initializers, regularizers=regularizers, automatic_diff=automatic_diff)

    def wake(self, use_natural=False, weights=None, global_step=None, layer_name="", imp_weights=None,
             fisher_calculator=None):
        if self._automatic_diff:
            self._gradients_w = self.auto_diff(loss=self.loss_w, variables=self.variables_w,
                                                            stops=[self.sample_w_q, self.inputs_r],
                                                            weights=weights)
        else:
            self._gradients_w = self.manual_diff(self.inputs_r, self.probs_w_p, self.sample_w_q,
                                                              weights=weights)

        if use_natural:
            self._natural_gradients_w = self.gen_distr._apply_fisher_multipliers(
                fisher_calculator=fisher_calculator,
                next_layer_distr_probs=self.probs_s_p,
                previous_layer_sample=self.inputs_p,
                grads=self._gradients_w,
                global_step=global_step,
                layer=layer_name,
                imp_weights=imp_weights)  # only one that needs this
            return self._natural_gradients_w, self.variables_w
        else:
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

        if use_natural:
            self._natural_gradients_ws = self.rec_distr._apply_fisher_multipliers(
                fisher_calculator=fisher_calculator,
                next_layer_distr_probs=self.probs_w_q,
                previous_layer_sample=self.inputs_r,
                grads=self._gradients_ws,
                global_step=global_step,
                layer=layer_name,
                imp_weights=None)
            return self._natural_gradients_ws, self.variables_s
        else:
            return self._gradients_ws, self.variables_s

    def sleep(self, use_natural=False, weights=None, global_step=None, layer_name="", fisher_calculator=None):
        if self._automatic_diff:
            self._gradients_s = self.auto_diff(loss=self.loss_s, variables=self.variables_s,
                                                            stops=[self.sample_s_p, self.inputs_p],
                                                            weights=weights)
        else:
            self._gradients_s = self.manual_diff(self.inputs_p, self.probs_s_q, self.sample_s_p,
                                                              weights=weights)
        if use_natural:
            self._natural_gradients_s= self.rec_distr._apply_fisher_multipliers(
                fisher_calculator=fisher_calculator,
                next_layer_distr_probs=self.probs_w_q,
                previous_layer_sample=self.inputs_r,
                grads=self._gradients_s,
                global_step=global_step,
                layer=layer_name,
                imp_weights=None)
            return self._natural_gradients_s, self.variables_s
        else:
            return self._gradients_s, self.variables_s
