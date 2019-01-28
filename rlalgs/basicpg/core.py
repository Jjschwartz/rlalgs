import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete
import .core


def get_action_dim(action_space):
    if isinstance(action_space, Box):
        return action_space.shape[0]
    elif isinstance(action_space, Discrete):
        return action_space.n
    raise NotImplementedError


def placeholder_from_space(space):
    if isinstance(space, Discrete):
        return tf.placeholder(tf.int32, shape=(None,))
    if isinstance(space, Box):
        tf.placeholder(tf.float32, shape=combined_shape(None, space.shape))


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)   # accepts tensor of any shape
    if np.isscalar(shape):
        return (length, shape)
    else:
        return (length, *shape)     # i.e. >=3D tuple e.g. (None, 3, 1)


def mlp(x, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    for size in hidden_sizes[:-1]:
        x = tf.layers.dense(x, size, activation=activation)
    return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)
