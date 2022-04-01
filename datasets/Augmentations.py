import numpy as np
import tensorflow as tf


def adjust_brightness(images, max_delta):
    delta = tf.random_uniform([1], -max_delta, max_delta, dtype=tf.float32)  # shape, min, max
    if isinstance(images, list):
        return tuple([tf.image.adjust_brightness(image, delta) for image in images])
    else:
        return tf.image.adjust_brightness(images, delta)


def adjust_brightnessAsym(images, min_delta, max_delta, mask=0):
    delta = tf.random_uniform([1], min_delta, max_delta, dtype=tf.float32)  # shape, min, max

    if mask == 1:

        if isinstance(images, list):

            raise Exception("TODO")
            return tuple([tf.image.adjust_brightness(image, delta) for image in images])
        else:

            # do not apply the transformation to the mask

            init_dims = [0 for s in images.shape[:-1].as_list()]
            end_dims = [-1 for s in images.shape[:-1].as_list()]

            mask = tf.slice(images, init_dims + [0], end_dims + [1])
            image = tf.slice(images, init_dims + [1], end_dims + [-1])

            image = tf.image.adjust_brightness(image, delta)

            return tf.concat([mask, image], axis=-1)
    else:
        if isinstance(images, list):
            return tuple([tf.image.adjust_brightness(image, delta) for image in images])
        else:
            return tf.image.adjust_brightness(images, delta)


def flip_up_down(images):
    if isinstance(images, list):
        return tuple([tf.image.flip_up_down(image) for image in images])
    else:
        return tf.image.flip_up_down(images)


def flip_left_right(images):
    if isinstance(images, list):
        return tuple([tf.image.flip_left_right(image) for image in images])
    else:
        return tf.image.flip_left_right(images)


def rot90(images):
    if isinstance(images, list):
        return tuple([tf.image.rot90(image) for image in images])
    else:
        return tf.image.rot90(images)


def sample_from_bernoulli(images):
    if isinstance(images, list):
        return tuple(
            [tf.distributions.Bernoulli(probs=(image + 1.0) / 2.0, dtype=image.dtype).sample() * 2.0 - 1.0 for image in
             images])
    else:
        return tf.distributions.Bernoulli(probs=(images + 1.0) / 2.0, dtype=images.dtype).sample() * 2.0 - 1.0


def get_id(pert):
    method, kwargs_tuple, frequency_perturbation = pert

    if method == "sample_from_bernoulli":
        assert (
            np.equal(frequency_perturbation, 1.0)), "Are you sure you don't want the frequency_perturbation to be 1.0?"
        return 'B' + 'f' + str(frequency_perturbation)
    elif method == "adjust_brightnessAsym":
        return 'ABm{}M{}f{}'.format(kwargs_tuple['min_delta'], kwargs_tuple['max_delta'], frequency_perturbation)
    elif method == "flip_up_down":
        return 'FUDf' + str(frequency_perturbation)
    else:
        raise Exception("perturbation not recognized: " + method)
