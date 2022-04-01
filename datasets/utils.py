import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np

def tf_sample_discrete_from_continuous(X):
    raise Exception("Are you sure you want to use this with the new 0/1 encoding?")
    return tfp.distributions.Bernoulli(probs=X, dtype=X.dtype).sample()

def tf_add_noise_to_discrete(X, flip_probability):
    probs = tf.ones(tf.shape(X)) * (1-flip_probability)
    flip = tfp.distributions.Bernoulli(probs=probs, dtype=X.dtype).sample()
    #flipped = bitwise_xor(tf.cast(X, dtype=tf.int32), tf.cast(flip, dtype=tf.int32))
    flipped = X * (2*flip-1)
    return flipped

'''
def add_gaussian_noise_and_clip(data, variance=0.1, low=0., high=1.):
    noise = np.random.normal(0, variance, size=X.shape)
    noisy_data = data + noise
    # clip in [low,high]
    noisy_data_clipped = np.clip(noisy_data, low, high)
    return noisy_X_clipped, noise
'''

def tf_add_gaussian_noise_and_clip(data, std=0.1, low=-1., high=1., clip_bool=True):
    # fixed problem, here it was variance, but everybody thought it was std... std is better to control since we understand the 'range' of the noise
    noise = tfp.distributions.Normal(tf.zeros_like(data), tf.ones_like(data) * std).sample()
    noisy_data = data + noise
    # clip in [low,high]
    if clip_bool:
        noisy_data = tf.clip_by_value(noisy_data, low, high)
    return noisy_data#, noise


def normalize(tensor, min_orig, max_orig, min_out=-1., max_out=1.):
    delta = max_out - min_out
    return delta * (tensor - min_orig) / (max_orig - min_orig) + min_out

def min_max_data_np(arrays):
    all_max = []
    all_min = []

    for arr in arrays:
        all_min.append(np.min(arr))
        all_max.append(np.max(arr))

    data_min = np.min(all_min)
    data_max = np.max(all_max)

    return data_min, data_max

