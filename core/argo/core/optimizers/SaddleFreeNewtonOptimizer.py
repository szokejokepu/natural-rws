from collections import namedtuple
import tensorflow as tf
import numpy as np

from tensorflow.python.ops.parallel_for import jacobian

from core.FFNetwork import FFNetwork
from core.argo.core.CostFunctions import CostFunctions


class SaddleFreeNewtonOptimizer(tf.train.GradientDescentOptimizer):
    """
    TODO: This doc
    """

    def __init__(self, damping_values, *args, **kw):

        self._model = kw["model"]

        # remove from args before passing to the constructor of tf.train.GradientDescentOptimizer
        kw.pop("model", None)

        # Use the default name if not specified.
        if 'name' not in kw:
            kw['name'] = 'SaddleFreeNewton'

        super().__init__(*args, **kw)
        self._damping_values = damping_values

    def compute_gradients(self, loss,
                          var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None,
                          colocate_gradients_with_ops=False, grad_loss=None):
        if var_list is None:
            # Make sure we have the same order of variables in the default case.
            # We do this just to decouple from tf's internals.
            var_list = tf.trainable_variables()

        # Get the gradients
        gradients_and_vars = super().compute_gradients(loss, var_list, gate_gradients, aggregation_method,
                                                       colocate_gradients_with_ops, grad_loss)

        # Compute the hessian
        with tf.name_scope('OptimizerHessian'):
            hessian_op_grid = [[None for _ in var_list] for _ in var_list]
            for i, var_i in enumerate(var_list):
                for j, var_j in enumerate(var_list):
                    if j >= i:
                        # Hessian is symmetric - enforce this. TODO: should we enforce through average rather?
                        hessian_block = jacobian(tf.gradients(loss, var_i)[0], var_j, use_pfor=False)
                        flattened_shape = (np.prod(var_i.shape), np.prod(var_j.shape))
                        flattened_hessian_block = tf.reshape(hessian_block, shape=flattened_shape)

                        hessian_op_grid[i][j] = flattened_hessian_block
                        hessian_op_grid[j][i] = tf.transpose(flattened_hessian_block)

            # Pack the hessian blocks into one matrix.
            hessian = tf.concat([tf.concat(hessian_op_list, axis=1) for hessian_op_list in hessian_op_grid], axis=0)
            grad = tf.concat([tf.reshape(gradients_and_vars[i][0], shape=(-1, 1)) for i in range(len(var_list))],
                             axis=0)

            # Preprocess the hessian by taking the abs of its eigs.
            eig_values, eig_vectors = tf.linalg.eigh(hessian)
            eig_values = tf.abs(eig_values)

            param_tensor = tf.constant(self._damping_values)
            # TODO: get the dtype somehow more elgantly
            line_search_loss_values = tf.Variable(initial_value=np.zeros(len(self._damping_values)), dtype=tf.float32)

            original_vars = tf.trainable_variables('ff_network')

            # We define a while loop that will iterate over the possible parameter values.
            #   The loop will build an exact copy of the network, such that we force the evaluation of the new network
            #   using the parameter given by the loop iterator.
            def copy_network_apply_grad_and_evaluate_loss(loop_iterator):
                # Copy the network
                network_copy = FFNetwork(self._model._opts, 'sfn_copy')
                logits_copy = network_copy(self._model.x)

                cost_function_copy = CostFunctions.instantiate_cost_function(self._model._opts["cost_function"],
                                                                             module_path="prediction")

                # Assign current weight values to network.
                copy_vars = tf.trainable_variables('sfn_copy')
                variable_sync_ops = tf.group([tf.assign(v_copy, v) for v, v_copy in zip(original_vars, copy_vars)])

                with tf.control_dependencies([variable_sync_ops]):
                    # Compute the weight update using the parameter which depends on i
                    lambdas = tf.abs(eig_values) + param_tensor[loop_iterator]
                    hessian = tf.matmul(eig_vectors, tf.matmul(tf.diag(lambdas), tf.transpose(eig_vectors)))

                    # Invert the hessian and multiply by grads
                    update_direction = tf.matmul(tf.linalg.inv(hessian), grad)

                    # Extract the update direction for each variable and reshape it.
                    update_ops = []
                    current_slice_idx = 0
                    for i, var in enumerate(copy_vars):
                        var_size = np.prod(var.shape)
                        var_update_direction = update_direction[current_slice_idx: current_slice_idx + var_size]

                        # Save the update direction directly in the grads_and_vars.
                        update_ops.append(tf.assign_sub(var, tf.reshape(var_update_direction, shape=var.shape)))

                        # Update the slice index
                        current_slice_idx += var_size

                    with tf.control_dependencies(update_ops):
                        # Compute and define the loss of the new network and save its  value to a loss tensor that
                        #   contains all such losses.
                        DummyModel = namedtuple('DummyModel', 'logits y')
                        local_loss = tf.reduce_sum(cost_function_copy(DummyModel(logits_copy, self._model.y))[0])
                        tf.assign(line_search_loss_values[loop_iterator], local_loss)

                next_loop_iterator = tf.add(loop_iterator, 1)
                return next_loop_iterator

            while_cond = lambda i: tf.less(i, len(self._damping_values))
            while_body = copy_network_apply_grad_and_evaluate_loss
            while_i = tf.constant(0)
            while_op = tf.while_loop(cond=while_cond,
                                     body=while_body,
                                     loop_vars=[while_i])

            with tf.control_dependencies([while_op]):
                argmin_ix = tf.argmin(line_search_loss_values, axis=0)

            # TODO: refactor this...
            # Compute the weight update using the parameter which depends on i
            lambdas = tf.abs(eig_values) + param_tensor[argmin_ix]
            hessian = tf.matmul(eig_vectors, tf.matmul(tf.diag(lambdas), tf.transpose(eig_vectors)))

            # Invert the hessian and multiply by grads
            update_direction = tf.matmul(tf.linalg.inv(hessian), grad)

            # Extract the update direction for each variable and reshape it.
            grads_and_vars = []
            current_slice_idx = 0
            for i, var in enumerate(original_vars):
                var_size = np.prod(var.shape)
                var_update_direction = update_direction[current_slice_idx: current_slice_idx + var_size]

                # Save the update direction directly in the grads_and_vars.
                grads_and_vars.append((tf.reshape(var_update_direction, shape=var.shape), var))

                # Update the slice index
                current_slice_idx += var_size

        return grads_and_vars
