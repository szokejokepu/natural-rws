import numpy as np
import tensorflow as tf

from core.networks.nodes.AbstractHMNode import D_TYPE
from core.utils.linalg.straight_inverse import _true_fisher_inverse, _damped_fisher_inverse


class FisherCalculator():
    def __init__(self, d_p, diagonal_pad, diagonal_cond, n_reg, model, k_step_update, simple_alter=False):

        self._d_p = d_p
        self._diagonal_pad = diagonal_pad
        self._diagonal_cond = diagonal_cond

        self._n_reg = n_reg
        self._nat_reg = [tf.constant(0.0)]

        self._model = model

        self._k_step_update = k_step_update
        self._simple_alter = simple_alter
        self._saves = {}

    def _multiply_grads_by_fisher_inv(self, weight_concat, difference, weights, previous_layer, global_step, orig_dtype,
                                      layer, k_len):  # , choice=False):
        weights = tf.cast(weights, dtype=D_TYPE)
        weight_concat = tf.cast(weight_concat, dtype=D_TYPE)
        previous_layer = tf.cast(previous_layer, dtype=D_TYPE)
        alpha = tf.cast(self._diagonal_pad, dtype=D_TYPE)

        if self._d_p is None or self._d_p == 0.0:
            inv_fisher = self._alternate_node(
                value_fn=lambda: [_true_fisher_inverse(U=previous_layer, Q=weights)],
                shapes=[
                    [difference.shape.as_list()[0], difference.shape.as_list()[-1], difference.shape.as_list()[-1]]],
                var_names=[layer + "MIT"],
                global_step=global_step,
                dtype=previous_layer.dtype)

            inverse_x_dif = tf.einsum('lkn,ln->lk', inv_fisher, difference)
        else:
            def _calc_sh():
                if (k_len < difference.shape[-1]):  # ^ choice:

                    C_inv = tf.linalg.diag(1 / weights)

                    grads = difference

                    u = self._alternate_node(
                        value_fn=lambda: [previous_layer],
                        shapes=[[previous_layer.shape.as_list()[0], k_len]],
                        var_names=[layer + "U"],
                        global_step=global_step,
                        dtype=previous_layer.dtype)

                    v_T = tf.transpose(u, perm=[1, 0])

                    m_inner_inv = self._alternate_node(
                        value_fn=lambda: [tf.linalg.inv(alpha * C_inv + tf.einsum('ij,jk->ik', v_T, u))],
                        shapes=[[grads.shape.as_list()[0], k_len, k_len]],
                        var_names=[layer + "MII"],
                        global_step=global_step,
                        dtype=previous_layer.dtype)

                    M2 = tf.einsum('ij,lj->li', u,
                                   tf.einsum('lij,lj->li', m_inner_inv, tf.einsum('ik,lk->li', v_T, grads)))

                    inverse_x_dif = ((1.0 + alpha) / alpha) * (grads - M2)

                    inverse_x_dif.set_shape(grads.shape.as_list())
                else:
                    inv_fisher = self._alternate_node(
                        value_fn=lambda: [
                            _damped_fisher_inverse(U=previous_layer, Q=weights, alpha=alpha)],
                        shapes=[[difference.shape.as_list()[0], difference.shape.as_list()[-1],
                                 difference.shape.as_list()[-1]]],
                        var_names=[layer + "MI"],
                        global_step=global_step,
                        dtype=previous_layer.dtype)
                    inverse_x_dif = tf.einsum('lkn,ln->lk', inv_fisher, difference)
                return inverse_x_dif

            inverse_x_dif = self._damping_multiplier(_calc_sh, lambda: difference)
            inverse_x_dif = self._simple_alternate(lambda: inverse_x_dif, alternate_fn=lambda: difference,
                                                   global_step=global_step)

        self._get_natural_regs(H=previous_layer, Q=weights, W=weight_concat, orig_dtype=orig_dtype)
        inverse_x_dif = tf.cast(inverse_x_dif, dtype=orig_dtype)
        return inverse_x_dif

    # This is to alternate between fisher and saved fisher (False) OR fisher and no fisher (True)
    def _simple_alternate(self, node_fn, alternate_fn, global_step):
        def tru():
            return node_fn()

        def fal():
            return alternate_fn()

        if self._simple_alter:
            return tf.cond(self._global_step_cond(global_step=global_step), tru, fal)
        else:
            return tru()


    def _alternate_node(self, value_fn, shapes, var_names, dtype, global_step):

        self._register_node(shapes, var_names, dtype)

        def tru():
            return self._update_node(value_fn(), var_names)

        def fal():
            return self._get_node_by_varname(var_names)

        return tf.cond(self._global_step_cond(global_step=global_step), tru, fal)

    def _global_step_cond(self, global_step):
        if self._k_step_update <= 0:
            return tf.constant(True)
        return tf.logical_or(tf.equal(global_step, 0), tf.equal(global_step % self._k_step_update, 0))

    def _update_node(self, var, var_names, validate_shape=True):
        vars = []
        for v, v_n in zip(var, var_names):
            if self._check_node(v_n):
                vars.append(tf.compat.v1.assign(self._saves[v_n], v, validate_shape=validate_shape))
            else:
                raise ValueError("Var name is accessed before creation: '{}'".format(v_n))
        return tuple(vars)

    def _register_node(self, exp_shapes, var_names, dtype, validate_shape=True):
        for (exp_shape, var_name) in zip(exp_shapes, var_names):
            if self._check_node(var_name):
                raise ValueError("Var name is already created: '{}'".format(var_name))
            else:
                if validate_shape == True:
                    self._saves[var_name] = tf.Variable(np.zeros(exp_shape), expected_shape=exp_shape, dtype=dtype)
                else:
                    self._saves[var_name] = tf.Variable([], shape=None, validate_shape=False, dtype=dtype)

    def _check_node(self, var_name):
        return var_name in self._saves

    def _get_node_by_varname(self, var_name):
        vars = []
        for v in var_name:
            if self._check_node(v):
                vars.append(self._saves[v])
            else:
                raise ValueError("Var name is accessed before assignment: '{}'".format(v))
        return tuple(vars)

    def _damping_multiplier(self, node_fn, alternate_fn):
        def tru():
            return node_fn()

        def fal():
            return alternate_fn()

        nody = tf.cond(self._diagonal_cond, tru, fal)
        return nody

    def _get_natural_regs(self, H, Q, W, orig_dtype):
        if self._n_reg > 0:
            W_T = tf.transpose(W, perm=[1, 0])

            regs_2 = tf.math.multiply(tf.math.sqrt(Q), tf.einsum('lk,kn->ln', W_T, H))
            # wHtQ/2 square sum
            loss = 0.0 + self._n_reg * tf.reduce_sum(tf.math.square(regs_2), name="nat_reg")
            self._nat_reg.append(tf.cast(loss, dtype=orig_dtype))

    def _get_natural_regularizers(self, weights):
        regs = [0.0] * len(weights)
        if self._n_reg > 0:
            loss = 0.0 + tf.add_n(self._nat_reg, name="natural_regularization")
            regs = tf.gradients(loss, weights)
        return regs
