import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import NadamOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

from core.HM2 import PHASE_WAKE, PHASE_SLEEP
from core.argo.core.optimizers.NesterovConst import NesterovConst


class WakeSleepOptimizer2(tf.compat.v1.train.GradientDescentOptimizer):

    def __init__(self, **optimizer_kwargs):
        self._model = optimizer_kwargs["model"]

        self._individual_learning_rate = optimizer_kwargs["individual_learning_rate"]

        self._learning_rate = optimizer_kwargs["learning_rate"]
        self._rescale_learning_rate = optimizer_kwargs["rescale_learning_rate"]
        self._d_p = None
        self._n_reg = None

        post_optimizer = optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs else None
        if post_optimizer is None or post_optimizer == "GD":
            self._post_optimizer = super()

        elif post_optimizer == "Momentum":
            self._post_optimizer = MomentumOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                     momentum=0.95,
                                                     use_locking=False,
                                                     name="MomentumOptimizer")

        elif post_optimizer == "RMSProp":
            self._post_optimizer = RMSPropOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                    decay=0.9,
                                                    epsilon=1e-5,
                                                    use_locking=False,
                                                    name="RMSPropOptimizer")

        elif post_optimizer == "Adam":
            self._post_optimizer = AdamOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                 beta1=0.9,
                                                 beta2=0.999,
                                                 epsilon=1e-8,
                                                 use_locking=False,
                                                 name="AdamOptimizer")
        elif post_optimizer == "Nadam":
            self._post_optimizer = NadamOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                  beta1=0.9,
                                                  beta2=0.999,
                                                  epsilon=1e-8,
                                                  use_locking=False,
                                                  name="NadamOptimizer")

        elif post_optimizer == "Nesterov":
            self._post_optimizer = MomentumOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                     momentum=0.95,
                                                     use_locking=False,
                                                     use_nesterov=True,
                                                     name="NesterovMomentumOptimizer")
        elif post_optimizer == "NesterovConst":
            self._post_optimizer = NesterovConst(model=self._model,
                                                 learning_rate=optimizer_kwargs["learning_rate"],
                                                 use_locking=False,
                                                 name="NesterovConstOptimizer")

        else:
            raise Exception("There is no such post optimizer defined. Must be: None, Adam, Momentum, RMSProp ...")

        super().__init__(self._learning_rate)
        self._network = self._model._network

        self._ilr = self.check_ilr(self._individual_learning_rate)

    def check_ilr(self, individual_learning_rate):
        length_of_network = len(self._network._layers_spec) + 1
        if isinstance(individual_learning_rate, list):
            assert len(individual_learning_rate) == length_of_network, \
                "Individual learning rates have to equal in length the number of layers, {} and {}".format(
                    individual_learning_rate, length_of_network)
            return list(map(float, individual_learning_rate))
        elif isinstance(individual_learning_rate, dict):
            ilr = [float(individual_learning_rate[i]) if i in individual_learning_rate else 1.0 for i in
                   range(length_of_network)]
            return ilr
        elif isinstance(individual_learning_rate, (int, float)):
            return [float(individual_learning_rate)] * length_of_network
        else:
            raise Exception("You gave an unexpected data type as Individual learning rates")

    def apply_gradients(self, grads_and_vars, global_step=None, name="WSOM"):
        return self._post_optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step, name=name)

    def compute_gradients(self, phase, *args, **kw):
        grads = []
        vars = []

        layers = self._network._layers
        if phase == PHASE_WAKE:
            for i, layer in enumerate(layers):
                grad, var = layer.wake()
                vars += list(var)
                if len(grad) > 0:
                    grads += [g * self._ilr[i] for g in grad]

        elif phase == PHASE_SLEEP:
            # CLASSIC SLEEP
            for i, layer in enumerate(layers[:-1]):
                grad, var = layer.sleep()
                vars += list(var)
                if len(grad) > 0:
                    grads += [g * self._ilr[i] for g in grad]

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

    def _get_regularizers(self, weights):
        regs = [0.0] * len(weights)
        if self._model.regularizers:
            loss = 0.0 + tf.add_n(self._model.regularizers, name="regularization")
            regs = tf.gradients(loss, weights)
        return regs
