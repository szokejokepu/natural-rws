import tensorflow as tf
import sonnet as snt
import numpy as np

from .AbstractModule import AbstractModule

import pdb

class Concatenate(AbstractModule):
    
    def __init__(self, node_name, channel_wise=False, name='Concatenate'):

        super().__init__(name=name)

        self._node_name = node_name
        self._channel_wise = channel_wise
        
    def _build(self, inputs):

        node = tf.get_default_graph().get_tensor_by_name(self._node_name + ":0")
        n_replicate = tf.shape(inputs)[0]/tf.shape(node)[0]
        
        if self._channel_wise:
            shape_tile = [1]*len(inputs.get_shape())
            shape_tile[0] = n_replicate
            
            img_x = inputs.get_shape()[1]
            img_y = inputs.get_shape()[2]
            node_image_like = snt.Linear(img_x * img_y)(node)
            node_to_be_concatenated = tf.tile(tf.reshape(node_image_like, [-1, img_x, img_y, 1]), shape_tile) 
        else:
            shape_tile = [1]*len(node.get_shape())
            shape_tile[0] = n_replicate
    
            node_to_be_concatenated = tf.tile(node, shape_tile)

        #node_to_be_concatenated
        return tf.concat([inputs, inputs], axis=-1)
