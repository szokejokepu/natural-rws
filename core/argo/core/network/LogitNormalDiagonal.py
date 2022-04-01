import tensorflow as tf
import sonnet as snt

import numpy as np

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from .AbstractGaussian import AbstractGaussian

import types

class LogitNormalDiagonal(AbstractGaussian):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0,
                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 clip_value = 0.001,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='logit_normal_diagonal'):

        super().__init__(module_tuple = module_tuple,
                         output_size = output_size,
                         output_shape = output_shape,
                         minimal_covariance = minimal_covariance,
                         covariance_parameterization=covariance_parameterization,
                         scalar_covariance = scalar_covariance,
                         initializers = initializers,
                         regularizers = regularizers,
                         contractive_regularizer = contractive_regularizer,
                         name = name)

        self._clip_value = clip_value
        
    def _build(self, inputs):
        mean, covariance, scale = self.create_mean_n_cov_layers(inputs)

        #TODO is this the kind of regularization we want. I think it makes sense.
        self.set_contractive_regularizer(mean, covariance,
                                        self._contractive_regularizer_inputs,
                                        self._contractive_regularizer_tuple,
                                        self._contractive_collection_network_str)
        
        gaussian = tfd.Normal(loc=mean, scale=scale)
        
        sigmoid_bijector = tfb.Sigmoid()
        logitnormal = tfd.TransformedDistribution(distribution = gaussian, bijector = sigmoid_bijector)
        
        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            # this is because there is not been for the LogitNormalDiagonal distribution
            return sigmoid_bijector.forward(gaussian.mean())
        
        logitnormal.reconstruction_node = types.MethodType(reconstruction_node, logitnormal)

        clip_value = self._clip_value
         
        # make sure a rescale the input for log_prob
        def log_prob(self, x, name='log_prob', **kwargs):
            # kinda of dirty I know, it is used to avoid recursion (****)
            return self._call_log_prob(tf_clip(x, low=-1.0 + clip_value, high=1.0 -clip_value), name=name, **kwargs)
        
        logitnormal.log_prob = types.MethodType(log_prob, logitnormal)

        
        return logitnormal
