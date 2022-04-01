import tensorflow as tf

from core.argo.core.network.AbstractModule import AbstractModule


class HMLogJointLikelihoodRWS2(AbstractModule):

    def __init__(self, name="HMLJLRWS"):
        super().__init__(name=name)

    def create_id(self, cost_fuction_kwargs):

        _id = "HMLJLRWS"

        return _id

    def _build(self, hm):
        n_samples = hm.n_z_samples

        log_joint_likelihood_p = hm._network._get_reconstruction_log_joint_likelihood_p()

        log_joint_likelihood_q = hm._network._get_dream_rec_log_joint_conditional_q()

        log_px_unnormalized = hm.importance_sampling_node - tf.math.log(tf.cast(n_samples, dtype=tf.float32))

        total_loss = -tf.reduce_mean(log_px_unnormalized, axis=0)

        reconstruction_loss = log_joint_likelihood_p
        dream_reconstruction_loss = log_joint_likelihood_q

        return total_loss, [[reconstruction_loss], [dream_reconstruction_loss]], [["NLL_X"], ["NLL_H"]], [{
            "fileName": "reconstruction_loss_NLLX"}, {
            "fileName": "reconstruction_loss_NLLH"}]
