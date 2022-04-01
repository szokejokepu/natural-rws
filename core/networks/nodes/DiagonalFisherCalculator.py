import tensorflow as tf

from core.networks.nodes.AbstractHMNode import D_TYPE
from core.networks.nodes.FisherCalculator import FisherCalculator
from core.utils.linalg.straight_trace import _true_fisher_inverse_trace, _damped_fisher_inverse_trace


class DiagonalFisherCalculator(FisherCalculator):

    def _multiply_grads_by_fisher_inv(self, weight_concat, difference, weights, previous_layer, global_step, orig_dtype,
                                      layer, k_len):
        weights = tf.cast(weights, dtype=D_TYPE)
        weight_concat = tf.cast(weight_concat, dtype=D_TYPE)
        previous_layer = tf.cast(previous_layer, dtype=D_TYPE)
        alpha = tf.cast(self._diagonal_pad, dtype=D_TYPE)

        if self._d_p is None or self._d_p == 0.0:
            inv_fisher = self._alternate_node(
                value_fn=lambda: [_true_fisher_inverse_trace(U=previous_layer, Q=weights)],
                shapes=[
                    [difference.shape.as_list()[0], difference.shape.as_list()[-1]]],
                var_names=[layer + "MIT"],
                global_step=global_step,
                dtype=previous_layer.dtype)

            inverse_x_dif = tf.einsum('ln,ln->ln', inv_fisher, difference)
        else:
            def _calc_sh():

                inv_fisher = self._alternate_node(
                    value_fn=lambda: [
                        _damped_fisher_inverse_trace(U=previous_layer, Q=weights, alpha=alpha)],
                    shapes=[[difference.shape.as_list()[0],
                             difference.shape.as_list()[-1]]],
                    var_names=[layer + "MI"],
                    global_step=global_step,
                    dtype=previous_layer.dtype)
                inverse_x_dif = tf.einsum('ln,ln->ln', inv_fisher, difference)
                return inverse_x_dif

            inverse_x_dif = self._damping_multiplier(_calc_sh, lambda: difference)
            inverse_x_dif = self._simple_alternate(lambda: inverse_x_dif, alternate_fn=lambda: difference,
                                                   global_step=global_step)

        self._get_natural_regs(H=previous_layer, Q=weights, W=weight_concat, orig_dtype=orig_dtype)
        inverse_x_dif = tf.cast(inverse_x_dif, dtype=orig_dtype)
        return inverse_x_dif
