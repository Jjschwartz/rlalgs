"""
Implementation of Vanilla Policy Gradient Deep RL algorithm.abs

Based off of OpenAI spinning up tutorial.
"""
import numpy as np
import tensorflow as tf
from spinup.utils.logx import EpochLogger


def mlp(x, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    for size in hidden_sizes[:-1]:
        x = tf.layers.dense(x, size, activation=activation)
    return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)


class ReplayBuffer:
    """
    A buffer for storing trajectories (o, a, r)
    """

    def __init__(self, obs_dim, act_dim, buffer_size):
        self.obs_buf = np.zeros((obs_dim, buffer_size))
        self.act_buf = np.zeros((act_dim, buffer_size))
        self.rew_buf = np.zeros((1, buffer_size))
        self.max_size = buffer_size
        self.ptr = 0

    def store(self, o, a, r):
        """
        Store a step outcome (o, a, r) in the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.ptr += 1

    def get(self):
        """
        Return the stored trajectories and empty the buffer
        """
        self.ptr = 0
        return [self.obs_buf, self.act_buf, self.rew_buf]


def vpg(env_fn, seed=0, logger_kwargs=dict()):
    """
    Vanilla Policy Gradient

    Arguments:
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment

    seed : random seed

    logger_kwargs : dictionary of arguments for the logger
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    
