import numpy as np
import tensorflow as tf
from tensorflow.python.ops.parallel_for import jacobian


def increment_iterator_after_op(i, op):
    with tf.control_dependencies([op]):
        return tf.add(i, 1)


def isolate_weights(op, temp_weight_register_map):
    # Store weights to temp
    with tf.control_dependencies([tf.assign(temp_weight_register_map[var], var) for var in tf.trainable_variables()]):
        # Do the thing.
        with tf.control_dependencies([op]):
            # Return the op that groups together the ops that restore the initial values of the weights from the temp.
            return tf.group([tf.assign(var, temp_weight_register_map[var]) for var in tf.trainable_variables()])


def argmin_over_parameter(value_to_minimize, parameter, parameter_values, initial_min_value=1e10):
    # Tensor containing all parameter values, in order.
    parameter_tensors = tf.concat([[tf.constant(p)] for p in parameter_values], axis=0)

    # A dict mapping variables to their respective temporary register - used for evaluating things without
    #   damaging the weights.
    temp_map = {var: tf.Variable(var, trainable=False) for var in tf.trainable_variables()}

    # The tensors used for computing the min/argmin.
    best_param_so_far = tf.constant(parameter_values[0])
    best_value_so_far = tf.constant(initial_min_value)

    def argmin_loop_body(param_tensor, i):
        def return_new_min():
            return parameter, value_to_minimize

        def return_stale_min():
            return best_param_so_far, best_value_so_far

        # First we assign the param tensor its value, then we record it if its better than what we've seen so far.
        with tf.control_dependencies([tf.assign(parameter, param_tensor[i])]):
            with tf.control_dependencies([tf.print(parameter)]):
                best_param, best_value = tf.cond(tf.logical_or(tf.less(value_to_minimize, best_value_so_far),
                                                               tf.equal(i, 0)),
                                                 true_fn=return_new_min,
                                                 false_fn=return_stale_min)
                with tf.control_dependencies([tf.assign(best_param_so_far, best_param),
                                              tf.assign(best_value_so_far, best_value)]):
                    return tf.print([value_to_minimize, best_value_so_far, best_param_so_far])

    # The while loop of the thing - condition is about the int iterator.
    condition = lambda i: tf.less(i, len(parameter_values))
    # The body is isolating one round of the while loop and then it increments its integer iterator.
    body = lambda i: increment_iterator_after_op(i, isolate_weights(argmin_loop_body(parameter_tensors, i), temp_map))

    loop_op = tf.while_loop(condition, body, [tf.constant(0)])
    with tf.control_dependencies([loop_op]):
        # Loop op needs to happen before the assignment of the best param to the param.
        return tf.assign(parameter, best_param_so_far)


class NewtonMethodOptimizer(tf.train.GradientDescentOptimizer):
    """
    Implements Newton's Method via gradient descent - when computing gradients, we multiply them by the inverse
    of the hessian to obtain NM's update step. These transformed gradients will be passed around as the 'gradients',
    i.e., they are returned by compute_gradients.
    """

    def __init__(self, damping=0.1, *args, **kw):

        self._model = kw["model"]

        # remove from args before passing to the constructor of tf.train.GradientDescentOptimizer
        kw.pop("model", None)

        # Use the default name if not specified.
        if 'name' not in kw:
            kw['name'] = 'NewtonMethod'

        super().__init__(*args, **kw)
        self._damping_constant = damping

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
        with tf.name_scope('NewtonMethodHessian'):
            hessian_op_grid = [[None for _ in var_list] for _ in var_list]
            for i, var_i in enumerate(var_list):
                for j, var_j in enumerate(var_list):
                    if j >= i:
                        # Hessian is symmetric - enforce this.
                        hessian_block = jacobian(tf.gradients(loss, var_i)[0], var_j, use_pfor=False)
                        flattened_shape = (np.prod(var_i.shape), np.prod(var_j.shape))
                        flattened_hessian_block = tf.reshape(hessian_block, shape=flattened_shape)

                        hessian_op_grid[i][j] = flattened_hessian_block
                        hessian_op_grid[j][i] = tf.transpose(flattened_hessian_block)

            # Pack the hessian blocks into one matrix.
            hessian = tf.concat([tf.concat(hessian_op_list, axis=1) for hessian_op_list in hessian_op_grid], axis=0)
            grad = tf.concat([tf.reshape(gradients_and_vars[i][0], shape=(-1, 1)) for i in range(len(var_list))],
                             axis=0)

            # Dampen the hessian
            if self._damping_constant is not None:
                hessian += tf.linalg.diag(self._damping_constant * tf.ones(grad.shape[0]))

            # Invert the hessian and multiply by grads
            update_direction = tf.matmul(tf.linalg.inv(hessian), grad)

            # Extract the update direction for each variable and reshape it.
            current_slice_idx = 0
            for i, var in enumerate(var_list):
                var_size = np.prod(var.shape)
                var_update_direction = update_direction[current_slice_idx: current_slice_idx + var_size]

                # Save the update direction directly in the grads_and_vars.
                gradients_and_vars[i] = (tf.reshape(var_update_direction, shape=var.shape), var)

                # Update the slice index
                current_slice_idx += var_size

        return gradients_and_vars
