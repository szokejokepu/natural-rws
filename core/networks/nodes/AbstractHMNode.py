import tensorflow as tf

from core.argo.core.network.AbstractModule import AbstractModule

WAKE_PHASE = "WAKE"
SLEEP_PHASE = "SLEEP"

D_TYPE = tf.float32
# D_TYPE = tf.float16
# D_TYPE = tf.float64


def get_all_axis_except_the_first(input):
    return list(range(1, len(input.shape)))


class AbstractHMNode(AbstractModule):

    def __init__(self, b_size, n_z_samples, b_size_int, n_z_samples_int, clip_probs=0, pm=True, initializers={},
                 regularizers={}, automatic_diff=False, **kwargs):
        super(AbstractHMNode, self).__init__()

        self._initializers = initializers
        self._regularizers = regularizers
        self._clip_probs = clip_probs
        self._pm = pm

        self._automatic_diff = automatic_diff

        self.b_size = b_size
        self.n_z_samples = n_z_samples
        self.mega_batch_size = b_size * n_z_samples
        self.b_size_int = b_size_int
        self.n_z_samples_int = n_z_samples_int
        self.mega_batch_size_int = b_size_int * n_z_samples_int
