"""
Common general functions used by algorithm implementations
"""
import psutil
import datetime
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.keras.backend as K
from gym.spaces import Box, Discrete


def get_dim_from_space(space):
    """
    Get the dimensions of a observation or action from a gym.space

    Arguments:
        space : the gym.space

    Returns:
        dim : the number of elements in a single entry if the space
    """
    if isinstance(space, Box):
        return space.shape[0]
    elif isinstance(space, Discrete):
        return space.n
    raise NotImplementedError


def placeholder_from_space(space, obs_space=False, name=None):
    """
    Generate the correct tf.placeholder from a gym.space, with optional name

    Arguments:
        space : the gym.space
        obs_space : whether the space if the observation space or not
        name : name for the tf.placeholder (leave as None for no name)

    Returns:
        ph : the tf.placeholder for the space
    """
    if isinstance(space, Discrete):
        dim = get_dim_from_space(space)
        if obs_space:
            return get_placeholder(tf.float32, combined_shape(None, dim), name)
        return get_placeholder(tf.int32, (None, ), name)
    elif isinstance(space, Box):
        return get_placeholder(tf.float32, combined_shape(None, space.shape), name)
    raise NotImplementedError


def combined_shape(length, shape=None):
    """
    Combines a tensor length and a shape into a single shape tuple
    """
    if shape is None:
        return (length,)   # accepts tensor of any shape
    if np.isscalar(shape):
        return (length, shape)
    else:
        # unpack shape tuple
        return (length, *shape)     # noqa: E999


def get_placeholder(dtype, shape, name=None):
    """
    Returns a placeholder.

    Used to abstract underlying implementation (tf or K)
    """
    return K.placeholder(dtype=dtype, shape=shape, name=name)


def reward_to_go(rews):
    """
    Calculate the reward-to-go return for each step in a given episode
    """
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def get_current_mem_usage():
    """
    Gets the current memory usage of calling process in MiB
    """
    process = psutil.Process()
    return process.memory_info().rss / float(2**20)


def print_current_mem_usage():
    """
    Prints memory usage of current process to stdout
    """
    mem = get_current_mem_usage()
    output = "# Mem usage = {} MiB #".format(mem)
    print("\n" + "-" * len(output))
    print(output)
    print("-" * len(output) + "\n")


def training_time_left(current_epoch, total_epochs, epoch_time):
    """
    Get predicted remaining time for training

    Arguments:
        int current_epoch : current epoch number
        int total_epochs : total number of training epochs
        float epoch_time : time required for a single epoch, in seconds

    Returns:
        str time_rem : the training time remaining as hour:min:sec format
    """
    epochs_rem = total_epochs - current_epoch - 1
    time_rem = epochs_rem * epoch_time
    # round to remove microseconds
    return str(datetime.timedelta(seconds=round(time_rem)))
