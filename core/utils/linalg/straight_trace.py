import tensorflow as tf


def _true_fisher_inverse_trace(U, Q):
    U_SQ = tf.math.square(U)
    M = tf.einsum('lk,nk->lnk', Q, U_SQ)
    F = tf.reduce_sum(M, axis=-1)

    F_inv = tf.math.reciprocal(F)
    return F_inv


def _damped_fisher_inverse_trace(U, Q, alpha):
    U_SQ = tf.math.square(U)
    M = tf.einsum('lk,nk->lnk', Q, U_SQ)
    F = tf.reduce_sum(M, axis=-1)

    F_hat = alpha * tf.ones(F.shape.as_list(), dtype=F.dtype) + F

    F_inv = tf.math.reciprocal(F_hat)
    return F_inv


