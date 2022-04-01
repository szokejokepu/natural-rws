import numpy as np
import tensorflow as tf

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.optimizers2.WakeSleepOptimizer2 import WakeSleepOptimizer2


class ReweightedWakeSleepOptimizer2(WakeSleepOptimizer2):

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)

        self._b_size = self._model.b_size
        self._n_samples = self._model.n_z_samples

        self._sleep_balance = 0.5
        self._wake_q = 1.0 - self._sleep_balance
        self._sleep_q = self._sleep_balance

        self._qbaseline = tf.constant(0.)
        if optimizer_kwargs["q_baseline"]:
            self._qbaseline = tf.constant(1.) / tf.cast(self._n_samples, dtype=tf.float32)

        self.bidirectional = False

    def compute_gradients(self, phase, *args, **kw):
        grads = []
        vars = []

        layers = self._network._layers
        if phase == PHASE_WAKE:

            # CALC WEIGHT PART
            if self.bidirectional:
                imp_weights_p = self._network.importance_weights_bihm
            else:
                imp_weights_p = self._network.importance_weights
            imp_weights_q = imp_weights_p - self._qbaseline
            # END OF CALC WEIGHT PART

            for i, layer in enumerate(layers):
                grad, var = layer.wake(weights=imp_weights_p)
                vars += list(var)

                if len(grad) > 0:
                    grads += [g * self._ilr[i] for g in grad]

            # WAKE PHASE SLEEP
            for i, layer in enumerate(layers[:-1]):
                grad, var = layer.wake_phase_sleep(weights=imp_weights_q)
                vars += list(var)

                if len(grad) > 0:
                    grads += [self._wake_q * g * self._ilr[i] for g in grad]
        elif phase == PHASE_SLEEP:
            # CLASSIC SLEEP
            if not self.bidirectional:
                for i, layer in enumerate(layers[:-1]):
                    grad, var = layer.sleep()
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

        grads_and_vars_not_none = [(tf.multiply(-lr, g, name="g_" + g.name.split(":")[0]) + r, v) for (g, r, v) in
                                   zip(grads, regs, vars) if g is not None]

        assert np.all([g.shape == v.shape for (g, v) in
                       grads_and_vars_not_none]), "The shapes of weights and gradients are not the same"

        return grads_and_vars_not_none
