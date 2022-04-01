import tensorflow as tf

from core.argo.core.network.AbstractModule import AbstractModule


class ClsLogJointLikelihood(AbstractModule):

    def __init__(self, name="CLSLJL"):
        super().__init__(name=name)

    def create_id(self, cost_fuction_kwargs):

        _id = "HMLJLRWS"

        return _id

    def _build(self, hm):
        total_loss, last_layer_ll = hm._network._get_log_joint_likelihood_p()

        return total_loss, last_layer_ll
