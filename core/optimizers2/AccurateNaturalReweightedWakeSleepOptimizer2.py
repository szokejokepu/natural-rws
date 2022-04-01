import numpy as np
import tensorflow as tf

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.optimizers2.NaturalReweightedWakeSleepOptimizerAlternate2 import NaturalReweightedWakeSleepOptimizerAlternate2


class AccurateNaturalReweightedWakeSleepOptimizer2(NaturalReweightedWakeSleepOptimizerAlternate2):

    def compute_gradients(self, phase, *args, global_step=None, **kw):
        grads = []
        vars = []

        self._nat_reg = [tf.constant(0.0)]
        layers = self._network._layers

        if phase == PHASE_WAKE:

            # CALC WEIGHT PART
            if self.bidirectional:
                imp_weights_p = self._network.importance_weights_bihm
            else:
                imp_weights_p = self._network.importance_weights
            imp_weights_q = imp_weights_p - self._qbaseline
            # imp_weights_q_2 = self.compute_accurate_normalized_weights(hr=hrw, hg=hgw)
            # END OF CALC WEIGHT PART

            for i, layer in enumerate(layers):
                layer_name = "W{}".format(str(i))
                grad, var = layer.wake(use_natural=(False if i in self._ignore_layers else True), weights=imp_weights_p, global_step=global_step,
                                       layer_name=layer_name,
                                       imp_weights=imp_weights_p,
                                       fisher_calculator=self.get_fisher_calculator())
                vars += list(var)
                if len(grad) > 0:
                    grads += [g * self._ilr[i] for g in grad]

            # WAKE PHASE SLEEP
            for i, layer in enumerate(layers[:-1]):
                layer_name = "WS{}".format(str(i))
                grad, var = layer.wake_phase_sleep(weights=imp_weights_q, use_natural=(False if i in self._ignore_layers else True), global_step=global_step,
                                                   layer_name=layer_name,
                                                   fisher_calculator=self.get_fisher_calculator())
                vars += list(var)

                if len(grad) > 0:
                    grads += [self._wake_q * g * self._ilr[i] for g in grad]

        elif phase == PHASE_SLEEP:
            # CLASSIC SLEEP
            if not self.bidirectional:
                for i, layer in enumerate(layers[:-1]):
                    layer_name = "S{}".format(str(i))
                    grad, var = layer.sleep(use_natural=(False if i in self._ignore_layers else True), global_step=global_step, layer_name=layer_name,
                                            fisher_calculator=self.get_fisher_calculator())
                    vars += list(var)
                    if len(grad) > 0:
                        grads += [self._sleep_q * g * self._ilr[i] for g in grad]

            else:
                # We have to do at least onceoperation which has grad 0 for simplicity
                grad, var = layers[-2].sleep()
                vars += list(var)

                if len(grad) > 0:
                    grads += [0.0 * g for g in grad]
        else:
            raise ValueError("invalid value for phase '{}'".format(phase))

        lr = 1.
        if phase == PHASE_SLEEP:
            lr *= self._rescale_learning_rate

        regs = self._get_regularizers(vars)
        nat_regs = self.get_fisher_calculator()._get_natural_regularizers(vars)
        grads_and_vars_not_none = [(tf.multiply(-lr, g, name="g_" + g.name.split(":")[0]) + r + nr, v) for (g, r, nr, v)
                                   in zip(grads, regs, nat_regs, vars) if g is not None]

        assert np.all([g.shape == v.shape for (g, v) in
                       grads_and_vars_not_none]), "The shapes of weights and gradients are not the same"

        return grads_and_vars_not_none

    def get_unnormalized_weigth_log(self, log_probs_p, log_probs_q):
        return log_probs_p - log_probs_q

    def compute_accurate_normalized_weights(self, hr, hg):
        with tf.name_scope("reweights"):
            log_probs_p = 0.0
            log_probs_q = 0.0

            for i in range(len(hg)):
                samples_q = hr[i][1]

                distr_p = hg[i][0]
                log_probs_p += tf.reduce_sum(distr_p.log_prob(samples_q), axis=-1)

                distr_q = hr[i][0]

                log_probs_q += tf.reduce_sum(distr_q.log_prob(samples_q), axis=-1) if i > 0 else tf.zeros(
                    tf.shape(samples_q)[0])

            # this is an intentional abuse of this function, would be too complicated to rewrite it
            unnormalized_weight_log = self.get_unnormalized_weigth_log(log_probs_q, log_probs_p)

            unnormalized_weight_log_reduced = tf.reduce_logsumexp(
                tf.reshape(unnormalized_weight_log, [self._n_samples, self._b_size]), axis=0)

            normalized_weights = tf.exp(
                unnormalized_weight_log - tf.tile(unnormalized_weight_log_reduced, [self._n_samples]))
        return normalized_weights
