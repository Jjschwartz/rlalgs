"""
This module contains functions for creating neural network models
"""
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
from tensorflow.keras import layers
from gym.spaces import Box, Discrete
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


def q_network(x, a, action_space, hidden_sizes=[64], activation=tf.nn.relu,
              output_activation=None):
    """
    Create a Q-network as a fully connected neural network, where the output
    layer is the q-value for each action in the action space

    Arguments:
        x : input placeholder
        a : action taken placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for output layer

    Returns:
        q_model : keras q network
        pi_fn : Keras functon for action selection
        q_pi : max q value for input
        act_q_val : the q value corresponding to action 'a' and input 'x'
    """
    act_dim = utils.get_dim_from_space(action_space)
    q_model = mlp(x, act_dim, hidden_sizes, activation, output_activation)

    pi = tf.squeeze(tf.argmax(q_model.output, axis=1))
    pi_fn = K.function(inputs=[x], outputs=[pi])

    q_pi = tf.reduce_max(q_model.output, axis=-1)
    action_mask = tf.one_hot(a, act_dim)
    act_q_val = tf.reduce_sum(action_mask * q_model.output, axis=-1)

    return q_model, pi_fn, q_pi, act_q_val


def mlp_actor_critic(x, a, action_space, hidden_sizes=[64], activation=tf.tanh,
                     output_activation=None):
    """
    Create fully-connected policy (actor) and value (critic) networks for a continuous or
    categorical policy.

    Arguments:
        x : input placeholder
        a : action taken placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        pi_model : keras policy network
        pi_fn : keras function for getting action for given input from policy network
        pi_logp : log probabilities of policy actions
        v_model : keras value network
        v_fn : keras function for getting value for given input from value network
    """
    if isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
    else:
        raise NotImplementedError

    act_dim = utils.get_dim_from_space(action_space)

    pi_model, pi_fn, pi_logp = policy(x, a, act_dim, hidden_sizes, activation, output_activation)
    v_model, v_fn = mlp_value_network(x, hidden_sizes, activation, output_activation)

    return pi_model, pi_fn, pi_logp, v_model, v_fn


def mlp_value_network(x, hidden_sizes=[32], activation=tf.tanh, output_activation=None):
    """
    Create a fully-connected value (critic) neural network.

    Arguments:
        x : input placeholder
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for output layer

    Returns:
        v_model : keras value network
        v_fn : keras function for getting value for given input
    """
    v_model = mlp(x, 1, hidden_sizes, activation, output_activation)
    v_predict = tf.squeeze(v_model.output, axis=1)
    v_fn = K.function(inputs=[x], outputs=[v_predict])
    return v_model, v_fn


def mlp_actor(x, a, action_space, hidden_sizes=[32], activation=tf.tanh,
              output_activation=None):
    """
    Create a fully-connected policy (actor) neural network for a continuous or categorical policy.

    Arguments:
        x : input placeholder
        a : action taken placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        model : keras policy model
        action_fn : action selection function
        log_probs : log probabilities tensor of policy actions
    """
    if isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
    else:
        raise NotImplementedError

    act_dim = utils.get_dim_from_space(action_space)
    return policy(x, a, act_dim, hidden_sizes, activation, output_activation)


def mlp_categorical_policy(x, a, act_dim, hidden_sizes=[64], activation=tf.tanh,
                           output_activation=None):
    """
    Create a fully-connected neural network for a categorical policy

    Arguments:
        x : input placeholder
        a : action taken placeholder
        act_dim : dimensions of action space
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        model : keras policy model
        action_fn : action selection function
        log_probs : log probabilities tensor of policy actions
    """
    model = mlp(x, act_dim, hidden_sizes, activation, output_activation)
    # log probs for calculating loss
    action_mask = K.one_hot(a, act_dim)
    log_probs = tf.reduce_sum(action_mask * tf.nn.log_softmax(model.output), axis=1)
    # action selection function
    act_predict = tf.squeeze(tf.random.categorical(model.output, 1), axis=1)
    action_fn = K.function(inputs=[x], outputs=[act_predict])
    return model, action_fn, log_probs


def mlp_gaussian_policy(x, a, act_dim, hidden_sizes=[64], activation=tf.tanh,
                        output_activation=None):
    """
    Create a fully-connected neural network for a continuous policy

    Arguments:
        x : input placeholder
        a : output placeholder
        act_dim : dimensions of action space
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        model : keras policy model
        action_fn : action selection function
        log_probs : log probabilities tensor of policy actions
    """
    model = mlp(x, act_dim, hidden_sizes, activation, output_activation)

    log_std = tf.Variable(-0.5*np.ones(act_dim, dtype=np.float32), trainable=False)
    log_probs = gaussian_likelihood(a, model.output, log_std)

    std = tf.exp(log_std)
    act_predict = model.output + tf.random.normal(tf.shape(model.output)) * std
    action_fn = K.function(inputs=[x], outputs=[act_predict])
    return model, action_fn, log_probs


def mlp(obs_ph, output_size, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    """
    Creates a fully connected neural network

    Arguments:
        obs_ph : K placeholder input to network
        output_size : number of neurons in output layer
        hidden_sizes : ordered list of size of each hidden layer
        activation : tf or K activation function for hidden layers
        output_activation : tf or K activation function for output layer or None for linear activation

    Returns:
        model : tf.keras model
    """
    x = obs_ph
    for size in hidden_sizes:
        x = layers.Dense(size, activation=activation)(x)
    acts_ph = layers.Dense(output_size, activation=output_activation)(x)
    return Model(inputs=obs_ph, outputs=acts_ph)


def gaussian_likelihood(x, mu, log_std):
    """
    Calculate the gaussian log-Likelihood for actions

    Arguments:
        a : action sample tensor
        mu : mean tensor
        log_std : log std tensor

    Returns:
        log likelihood tensor
    """
    std = tf.exp(log_std)
    pre_sum = tf.square((x - mu)/std) + 2*log_std + np.log(2*np.pi)
    return -0.5 * tf.reduce_sum(pre_sum, axis=1)
