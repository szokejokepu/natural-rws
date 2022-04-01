import types
from operator import xor

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from .AbstractModule import AbstractModule
from .PlusMinusOneMapping import PlusMinusOneMapping


class BernoulliPlusMinusOne(AbstractModule):

    def __init__(self, output_size=-1, output_shape=-1, initializers={}, regularizers={}, clip_value=0, dtype=None,
                 name='Bernoulli'):

        super().__init__(name=name)

        assert xor(output_size == -1,
                   output_shape == -1), "Either output_size or output_shape mut be specified, not both"

        if output_size != -1:
            self._output_shape = [output_size]
        else:
            self._output_shape = output_shape

        self._initializers = initializers
        self._regularizers = regularizers

        self._clip_value = clip_value
        self._dtype = dtype

    def _build(self, inputs):
        # create the layers for mean and covariance
        output_shape = [-1] + self._output_shape
        logits = tf.reshape(
            snt.Linear(np.prod(self._output_shape), initializers=self._initializers, regularizers=self._regularizers)(
                inputs), output_shape)

        dtype = tf.float32  # inputs.dtype
        if self._dtype is not None:
            dtype = self._dtype

        if self._clip_value > 0:
            probs = tf.nn.sigmoid(logits)

            probs = tf.clip_by_value(probs, self._clip_value, 1 - self._clip_value)
            bernoulli = tfp.distributions.Bernoulli(probs=probs, dtype=dtype)
        else:
            bernoulli = tfp.distributions.Bernoulli(logits=logits, dtype=dtype)

        affine_transform = PlusMinusOneMapping(scale=2., shift=-1.)
        bernoulli_plus_minus_one = tfp.distributions.TransformedDistribution(distribution=bernoulli,
                                                                             bijector=affine_transform,
                                                                             name="BernoulliPlusMinusOne")

        def reconstruction_node(self):
            return self.mean()

        bernoulli_plus_minus_one.reconstruction_node = types.MethodType(reconstruction_node, bernoulli_plus_minus_one)

        def distribution_parameters(self):
            return [self.mean()]

        bernoulli_plus_minus_one.distribution_parameters = types.MethodType(distribution_parameters,
                                                                            bernoulli_plus_minus_one)

        def get_probs(self):
            return self.distribution.probs

        bernoulli_plus_minus_one.get_probs = types.MethodType(get_probs, bernoulli_plus_minus_one)

        def params(self):
            return (self.mean(),)

        bernoulli_plus_minus_one.params = types.MethodType(params, bernoulli_plus_minus_one)

        def default_prior(self, prior_shape):
            halfes = tf.ones(shape=prior_shape) / 2.0
            prior = tfp.distributions.Bernoulli(probs=halfes, name="prior", dtype=self._dtype)
            pm_prior = tfp.distributions.TransformedDistribution(distribution=prior, bijector=affine_transform,
                                                                 name="BernoulliPlusMinusOne")
            return pm_prior

        bernoulli_plus_minus_one.default_prior = types.MethodType(default_prior, bernoulli_plus_minus_one)

        def kl_divergence(self, other_transformed):
            return self.distribution.kl_divergence(other_transformed.distribution)

        bernoulli_plus_minus_one.kl_divergence = types.MethodType(kl_divergence, bernoulli_plus_minus_one)

        return bernoulli_plus_minus_one
