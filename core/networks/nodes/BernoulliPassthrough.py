import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Bernoulli

from core.argo.core.network.AbstractModule import AbstractModule
from core.argo.core.network.PlusMinusOneMapping import PlusMinusOneMapping

def get_all_axis_except_the_first(input):
    return list(range(1, len(input.shape)))

class BernoulliPassthrough(AbstractModule):
    def __init__(self, pm, clip_probs=0):
        super(BernoulliPassthrough, self).__init__()
        self._pm = pm
        self._clip_probs = clip_probs

    # @tf.function
    @snt.reuse_variables
    def _build(self, logits, **kwargs):
        if self._clip_probs > 0:
            probs = tf.nn.sigmoid(logits)

            probs = tf.clip_by_value(probs, self._clip_probs, 1 - self._clip_probs)

            self.distr_og = Bernoulli(probs=probs, dtype=probs.dtype)
        else:
            self.distr_og = Bernoulli(logits=logits, dtype=logits.dtype)

        if self._pm:
            affine_transform = PlusMinusOneMapping(scale=2., shift=-1.)
            self.distr = tfp.distributions.TransformedDistribution(distribution=self.distr_og,
                                                                   bijector=affine_transform,
                                                                   name="BernoulliPlusMinusOne")
        else:
            self.distr = self.distr_og

    def sample(self, size=()):
        return self.distr.sample(size)
        # return tf.grad_pass_through(self.distr.sample)(size)

    def log_prob(self, samples):
        return tf.reduce_sum(self.distr.log_prob(samples), axis=get_all_axis_except_the_first(samples))

    def prob(self, samples):
        return self.distr.prob(samples)

    @property
    def probs(self):
        return self.distr_og.probs

    def mean(self):
        return self.distr.mean()

    def shape(self):
        return tf.shape(self.distr.sample())
