import numpy as np
import tensorflow as tf

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.networks.nodes.AbstractHMNode import D_TYPE
from core.networks.nodes.DiagonalFisherCalculator import DiagonalFisherCalculator
from core.networks.nodes.FisherCalculator import FisherCalculator
from core.networks.nodes.LinSolveFisherCalculator import LinSolveFisherCalculator
from core.optimizers2.WakeSleepOptimizer2 import WakeSleepOptimizer2


def get_diagonal_pad(diagonal_pad):
    dp = None

    global_step = tf.compat.v1.train.get_or_create_global_step()
    if diagonal_pad is None:
        pass

    elif isinstance(diagonal_pad, (int, float)):
        dp = tf.constant(float(diagonal_pad), dtype=D_TYPE)

    elif isinstance(diagonal_pad, tuple):
        dp_min, dp_name, dp_kwargs = diagonal_pad
        dp_kwargs = dp_kwargs.copy()
        dp_method = getattr(tf.train, dp_name)
        dp_kwargs.update({
            "global_step": global_step})
        dp = dp_min + dp_method(**dp_kwargs)

    # instantiate lr node if lr is None and diagonal_pad is a dict at this point
    if dp is None and isinstance(diagonal_pad, dict):
        if not 0 in diagonal_pad:
            raise ValueError(
                "diagonal_pad schedule must specify, learning rate for step 0. Found schedule: %s" % diagonal_pad)

        dp = tf.constant(diagonal_pad[0])
        for key, value in diagonal_pad.items():
            dp = tf.cond(
                tf.less(global_step, key), lambda: dp, lambda: tf.constant(value))
        tf.summary.scalar("diagonal_pad", dp)

    if dp is None:
        raise Exception("oops, something went wrong... could not process diagonal_pad {}".format(str(diagonal_pad)))

    return tf.identity(dp, name="diagonal_pad")


class NaturalWakeSleepOptimizer2(WakeSleepOptimizer2):

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)

        self._ignore_layers = optimizer_kwargs["ignore_layers"] if "ignore_layers" in optimizer_kwargs else []
        self._d_p = optimizer_kwargs["diagonal_pad"]
        self._n_reg = (optimizer_kwargs["natural_reg"] if "natural_reg" in optimizer_kwargs and float(
            optimizer_kwargs["natural_reg"]) > 0 else 0)
        self._b_size = self._model.b_size
        self._n_samples = self._model.n_z_samples
        self._n_z_length = self._n_samples * self._b_size

        self._diagonal_pad = get_diagonal_pad(self._d_p)
        self._diagonal_cond = tf.less_equal(self._diagonal_pad, 10.0)
        self._k_step_update = 1

        self._diagonal = False
        self._linsolve = False

        self._simple_alter = (optimizer_kwargs["simple_alter"] if "simple_alter" in optimizer_kwargs else False)

    def get_fisher_calculator(self):
        if not hasattr(self, '_fisher_calculator'):
            if self._diagonal:
                self._fisher_calculator = DiagonalFisherCalculator(self._d_p, self._diagonal_pad, self._diagonal_cond,
                                                                   self._n_reg, self._model, self._k_step_update,
                                                                   self._simple_alter)
            else:
                if self._linsolve:
                    self._fisher_calculator = LinSolveFisherCalculator(self._d_p, self._diagonal_pad, self._diagonal_cond,
                                                               self._n_reg, self._model, self._k_step_update,
                                                               self._simple_alter)
                else:
                    self._fisher_calculator = FisherCalculator(self._d_p, self._diagonal_pad, self._diagonal_cond,
                                                               self._n_reg, self._model, self._k_step_update,
                                                               self._simple_alter)

        return self._fisher_calculator

    def compute_gradients(self, phase, *args, global_step=None, **kw):
        grads = []
        vars = []

        self._nat_reg = [tf.constant(0.0)]

        if phase == PHASE_WAKE:
            layers = self._network._layers
            for i, layer in enumerate(layers):
                layer_name = "W{}".format(str(i))
                grad, var = layer.wake(use_natural=(False if i in self._ignore_layers else True), global_step=global_step, layer_name=layer_name,
                                       fisher_calculator=self.get_fisher_calculator())
                vars += list(var)

                if len(grad) > 0:
                    grads += [g * self._ilr[i] for g in grad]

        elif phase == PHASE_SLEEP:

            layers = self._network._layers

            for i, layer in enumerate(layers[:-1]):
                layer_name = "S{}".format(str(i))
                grad, var = layer.sleep(use_natural=(False if i in self._ignore_layers else True), global_step=global_step, layer_name=layer_name,
                                        fisher_calculator=self.get_fisher_calculator())
                vars += list(var)
                if len(grad) > 0:
                    grads += [g * self._ilr[i] for g in grad]
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
