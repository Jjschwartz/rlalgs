"""
Core functions for use with Vanilla Policy Gradient (VPG) implementation
"""
import tensorflow as tf
import rlalgs.utils.utils as utils
from gym.spaces import Box, Discrete


def mlp_actor_critic(x, a, action_space, hidden_sizes=[64],
                     activation=tf.tanh, output_activation=None):
    """
    Create the actor critic model of VPG

    Arguments:
        x : tf placeholder input to network
        a : tf placeholder for actions
        action_space : action space of environment as gym.space
        hidden_sizes : ordered list of size of each hidden layer
        activation : tf activation function for hidden layers
        output_activation : tf activation function for output layer or None if no activation

    Returns:
        pi : policy network action selection as tf tensor
        logp : log probability of action in policy network as tf tensor
        v : output of value network as tf tensor
    """
    if isinstance(action_space, Box):
        policy = utils.mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = utils.mlp_categorical_policy
    else:
        raise NotImplementedError

    # create policy and value networks
    # create within scopes so to insure seperate models are created since we call the same
    # method to create both models
    with tf.variable_scope("pi"):
        pi, logp = policy(x, a, action_space, hidden_sizes, activation, output_activation)
    with tf.variable_scope("v"):
        v = tf.squeeze(utils.mlp(x, 1, hidden_sizes, activation, output_activation), axis=1)

    return pi, logp, v
