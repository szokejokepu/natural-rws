import tensorflow as tf

from core.optimizers2.ReweightedWakeSleepOptimizer2 import ReweightedWakeSleepOptimizer2


class BidirectionalHMOptimizer2(ReweightedWakeSleepOptimizer2):

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)

        self._b_size = self._model.b_size
        self._n_samples = self._model.n_z_samples

        self._sleep_balance = 0.0
        self._wake_q = 1.0 - self._sleep_balance
        self._sleep_q = self._sleep_balance

        self._qbaseline = tf.constant(0.)
        if optimizer_kwargs["q_baseline"]:
            self._qbaseline = tf.constant(1.) / tf.cast(self._n_samples, dtype=tf.float32)

        self.bidirectional = True
