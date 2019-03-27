"""
Core functions for Advantage Actor-Critic (A2C) implementation
"""
import tensorflow as tf
import rlalgs.utils.utils as utils
from gym.spaces import Box, Discrete


def mlp_actor_critic(x, a, action_space, hidden_sizes=[64], activation=tf.tanh,
                     output_activation=None):
    if isinstance(action_space, Box):
        policy = utils.mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = utils.mlp_categorical_policy
    else:
        raise NotImplementedError

    with tf.variable_scope("pi"):
        pi, pi_logp = policy(x, a, action_space, hidden_sizes, activation, output_activation)
    with tf.variable_scope("v"):
        v = tf.squeeze(utils.mlp(x, 1, hidden_sizes, activation, output_activation), axis=1)
    return pi, pi_logp, v
