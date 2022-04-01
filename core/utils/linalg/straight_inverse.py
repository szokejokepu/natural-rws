import tensorflow as tf


def _damped_fisher_inverse_adapt(U, Q, C):
    U_T = tf.transpose(U, perm=[1, 0])
    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    lambda_max = tf.linalg.svd(F, compute_uv=False)[:,0]
    alpha = tf.einsum("l,lik->lik",lambda_max / (C - 1), tf.eye(F.shape.as_list()[-1], batch_shape=lambda_max.shape.as_list(), dtype=F.dtype))
    F_hat = alpha + F

    F_inv = tf.linalg.inv(F_hat)
    return (1.0 + alpha) * F_inv


def _damped_fisher_inverse(U, Q, alpha):

    U_T = tf.transpose(U, perm=[1, 0])
    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    F_hat = alpha * tf.eye(F.shape.as_list()[-1], dtype=F.dtype) + F

    F_inv = tf.linalg.inv(F_hat)
    return (1.0 + alpha) * F_inv

def _damped_fisher(U, Q, alpha):

    U_T = tf.transpose(U, perm=[1, 0])
    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    F_hat = alpha * tf.eye(F.shape.as_list()[-1], dtype=F.dtype) + F

    return 1/(1.0 + alpha) * F_hat


def _true_fisher_inverse(U, Q):
    U_T = tf.transpose(U, perm=[1, 0])
    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    F_inv = tf.linalg.inv(F)
    return F_inv

def _true_fisher(U, Q):
    U_T = tf.transpose(U, perm=[1, 0])
    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    return F
