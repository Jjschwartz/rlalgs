"""
Some core functions for Deep Q-network implementation
"""
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils


def q_network(x, a, action_space, hidden_sizes=[64], activation=tf.tanh,
              output_activation=None):
    """
    Create a Q-network as a fully connected neural network, where the output
    layer is the q-value for each action in the action space

    Arguments:
        x : input placeholder
        a : output placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for output layer

    Returns:
        pi : action selection tensor (max{a} Q(s, a))
        q_val_max : the q value of best action
        act_q_val : the q value corresponding to a given action
    """
    act_dim = utils.get_dim_from_space(action_space)
    q_vals = utils.mlp(x, act_dim, hidden_sizes, activation, output_activation)
    pi = tf.argmax(q_vals, axis=1)
    q_val_max = tf.reduce_max(q_vals, axis=1)
    action_mask = tf.one_hot(a, act_dim)
    act_q_val = tf.reduce_sum(action_mask * q_vals, axis=1)
    return pi, q_val_max, act_q_val
