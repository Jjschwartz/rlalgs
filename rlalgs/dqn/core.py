"""
Some core functions for Deep Q-network implementation
"""
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils


def get_vars(scope):
    # scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return scope_vars


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def mlp(x, output_size, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    """
    Creates a fully connected neural network

    Arguments:
        x : tf placeholder input to network
        output_size : number of neurons in output layer
        hidden_sizes : ordered list of size of each hidden layer
        activation : tf activation function for hidden layers
        output_activation : tf activation function for output layer or None if no activation

    Returns:
        y : output layer as tf tensor
    """
    for size in hidden_sizes:
        x = tf.layers.dense(x, size, activation=activation)
    return tf.layers.dense(x, output_size, activation=output_activation)


def q_network(x, a, action_space, hidden_sizes=[64], activation=tf.nn.relu,
              output_activation=None):
    """
    Create a Q-network as a fully connected neural network, where the output
    layer is the q-value for each action in the action space

    Arguments:
        x : input placeholder or variable
        a : output action placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for output layer

    Returns:
        pi : action selection tensor (max{a} Q(s, a)) for input 'x'
        q_pi : q value of best action for input 'x'
        act_q_val : the q value corresponding to action 'a' and input 'x'
    """
    act_dim = utils.get_dim_from_space(action_space)
    q_vals = mlp(x, act_dim, hidden_sizes, activation, output_activation)
    pi = tf.squeeze(tf.argmax(q_vals, axis=1))
    q_pi = tf.reduce_max(q_vals, axis=-1)
    action_mask = tf.one_hot(a, act_dim)
    act_q_val = tf.reduce_sum(action_mask * q_vals, axis=-1)
    return pi, q_pi, act_q_val, q_vals
