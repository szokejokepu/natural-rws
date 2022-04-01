import tensorflow as tf

import numpy as np

import pdb

from abc import ABC, abstractmethod

from .utilsOptimizers import my_loss

from .DropOneLogitOptimizer import DropOneLogitOptimizer

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian, jacobian
        
class DropOneLogitGradientDescentOptimizer(tf.train.GradientDescentOptimizer, DropOneLogitOptimizer):

    def __init__(self, learning_rate, *args, **kw):

        self._model = kw["model"]

        # remove from args before passing to the constructor of tf.train.GradientDescentOptimizer
        kw.pop("model", None)

        if "name" not in kw.keys():
            kw["name"] = "DropOneLogitGradientDescent"
            
        super().__init__(learning_rate, *args, **kw)
        
    def compute_gradients(self, loss, *args, **kw):

        logits = self._model.logits
        y = self._model.y
        regularizer = self._model.regularizer

        ########################
        
        # probabilities
        logits_add_last_node = tf.concat([logits,
                                          tf.zeros_like(tf.slice(logits, [0, 0], [-1, 1]))],
                                         1)
        
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y, tf.int32), logits = logits_replace_last_node)
            

        # this works
        #probabilities = tf.nn.softmax(self.logits)

        # this doesnt work
        #probabilities = tf.nn.softmax(logits_replace_last_node)

        # this work!            
        new_loss = my_loss(y, logits_add_last_node)
        self.likelihood_per_sample = new_loss

        self._model.loss = new_loss
        self._model.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_add_last_node, axis=1),
                                                               tf.cast(y, dtype = tf.int64)),
                                                      dtype = tf.float32))
        
        total_loss = tf.reduce_mean(new_loss) + regularizer

        grads_and_vars = super().compute_gradients(total_loss, *args, **kw)

        # apply clipping
        #clipping_method, clipping_kwargs = self._grad_clipping_tuple
        
        grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]
        grads = [g for (g, v) in grads_and_vars_not_none]
        variables = [v for (g, v) in grads_and_vars_not_none]
        grads_and_vars = [(g, v) for (g, v) in zip(grads, variables)]
        
        return grads_and_vars

