import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np

import pdb

from .utils import tf_sample_discrete_from_continuous, tf_add_gaussian_noise_and_clip

def context_encoding(image, size, margin=0, color=1):
    #if isinstance(images, list):
    #    return tuple([image for image in images])
    #else:
    #pdb.set_trace()

    '''
    matrix = np.zeros([28, 28, 1])
    matrix[0:14, 0:14, :] = 1
    mask = tf.convert_to_tensor(matrix, dtype=tf.int32)

    indexes = tf.where(tf.equal(mask, 1))
    '''

    #indices = tf.constant([[0, 0, 0], [0, 1, 0] [1, 0, 0], [1, 1, 0]])

    w, h, c = image.get_shape().as_list()

    if c!=1:
        raise Exception("contextual encoding not implemented for color images, feel free to implement it")

    # np code
    '''
    x = np.random.randint(w - size)
    y = np.random.randint(h - size)
    
    rows, columns = np.meshgrid(range(x,x+size), range(y,y+size))

    channels = np.zeros(size**2, dtype=np.int32)

    indices = tf.stack([[int(r) for r in rows.flatten()], [int(c) for c in columns.flatten()], channels], axis=1)

    updates = np.ones(size**2, dtype=np.int32)*1 # white square, use -1 for black
    '''

    # tf code
    x_coord = tf.random.uniform(shape=[], maxval=w-size-2*margin, dtype=tf.int64)
    y_coord = tf.random.uniform(shape=[], maxval=h-size-2*margin, dtype=tf.int64)
    
    rows, columns = tf.meshgrid(tf.range(x_coord+margin, x_coord+margin + size), tf.range(y_coord+margin, y_coord+margin + size))
    flatten_rows = tf.reshape(rows, [-1])
    flatten_cols = tf.reshape(columns, [-1])

    channels = tf.zeros(size**2, dtype=tf.int64)
  
    indices = tf.transpose(tf.stack((flatten_rows, flatten_cols, channels))) 

    if color == 'r':
        updates = tf.ones(size**2)*tf.random.uniform([1], minval=-1, maxval=1)
    else:
        updates = tf.ones(size**2)*color

    ce_image = tf.tensor_scatter_nd_update(image, indices, updates)
    
    return ce_image


def gaussian_noise(image, std, clip=1):
    return tf_add_gaussian_noise_and_clip(image, std=std, clip_bool=clip)

def salt_and_pepper(image, delta, prob, clip=1):
    sign_flip = (tfp.distributions.Bernoulli(probs = tf.ones_like(image)*0.5, dtype=tf.float32).sample() * 2. - 1.)*tfp.distributions.Bernoulli(probs = tf.ones_like(image)*prob, dtype=tf.float32).sample()
    image = image + 2*delta*sign_flip
    
    if clip:
        image  = tf.clip_by_value(image, -1., 1.)
        
    return image
    
def get_id(pert):
    method, kwargs_tuple, frequency_perturbation = pert

    if method=="gaussian_noise":
        std = kwargs_tuple["std"]
        clip = kwargs_tuple["clip"]
        return 'GN' + 's' + str(std) + 'c' + str(clip) + 'f' + str(frequency_perturbation)

    elif method=="salt_and_pepper":
        delta = kwargs_tuple["delta"]
        clip = kwargs_tuple["clip"]
        prob = kwargs_tuple["prob"]
        return 'SP' + 'd' + str(delta) + 'p' + str(prob) + 'c' + str(clip) + 'f' + str(frequency_perturbation)

    elif method=="context_encoding":
        size = kwargs_tuple["size"]
        margin = kwargs_tuple["margin"]
        color = kwargs_tuple["color"]
        return 'CE' + 's' + str(size) + 'm' + str(margin) + 'c' + str(color) +'f' + str(frequency_perturbation)

    else:
        raise Exception("perturbation not recognized: " + method)
    
