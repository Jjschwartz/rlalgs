"""
This module contains functions for creating neural network models
"""
import numpy as np
from gym.spaces import Box, Discrete

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import rlalgs.utils.utils as utils
import rlalgs.algos.policy as policy_fn


def q_network(x, action_space, hidden_sizes=[64], activation=tf.nn.relu,
              output_activation=None):
    """
    Create a Q-network as a fully connected neural network, where the output
    layer is the q-value for each action in the action space

    Arguments:
        x : input placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for output layer

    Returns:
        q_model : keras q network
        pi_fn : action selection functon
        q_fn : q value function, which return max q value for input and
            q value corresponding to action 'a' and input 'x'
    """
    act_dim = utils.get_dim_from_space(action_space)
    q_model = mlp(x, act_dim, hidden_sizes, activation, output_activation)

    pi_fn = policy_fn.discrete_qlearning(q_model)

    @tf.function
    def q_fn(a, o):
        q_pi = tf.reduce_max(q_model(o), axis=-1)
        action_mask = tf.one_hot(a, act_dim)
        act_q_val = tf.reduce_sum(action_mask * q_pi, axis=-1)
        return q_pi, act_q_val

    return q_model, pi_fn, q_fn


def mlp_actor_critic(x, a, action_space, hidden_sizes=[64], activation=tf.tanh,
                     output_activation=None, share_layers=False):
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
        share_layers : whether to share common layers between actor and critic models or not

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
    pi_model, last_hidden_layer = mlp(x, x, act_dim, hidden_sizes, activation, output_activation)

    if share_layers:
        v_model, _ = mlp(x, last_hidden_layer, 1, [], activation, output_activation)
    else:
        v_model, _ = mlp(x, x, 1, hidden_sizes, activation, output_activation)

    pi_fn = policy(pi_model)
    v_fn = mlp_value_network(v_model)

    return pi_model, pi_fn, v_model, v_fn


def mlp_value_network(v_model):
    """
    Create functions for value (critic) neural network.

    Arguments:
        v_model : keras value network

    Returns:
        v_fn : keras function for getting value for given input
    """
    @tf.function
    def model_query(o):
        return tf.squeeze(v_model(o), axis=1)

    def v_fn(o):
        return model_query(o).numpy()
    return v_fn


def mlp_actor(x, action_space, hidden_sizes=[32], activation=tf.tanh, output_activation=None):
    """
    Create a fully-connected policy (actor) neural network for a continuous or categorical policy.

    Arguments:
        x : input placeholder
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
    pi_model, _ = mlp(x, x, act_dim, hidden_sizes, activation, output_activation)
    pi_fn = policy(pi_model)
    return pi_model, pi_fn


def mlp_categorical_policy(model):
    """
    Create functons for a categorical policy

    Arguments:
        model : keras policy model

    Returns:
        action_fn : action selection function
    """
    return policy_fn.discrete_pg(model)


def mlp_gaussian_policy(model):
    """
    Create functions for a continuous policy

    Arguments:
        model : keras policy model

    Returns:
        action_fn : action selection function
    """
    # log probs for calculating loss
    # log_std = tf.Variable(-0.5*np.ones(act_dim, dtype=np.float32), trainable=False)
    # log_probs = gaussian_likelihood(a, model.output, log_std)
    # action selection function
    # std = tf.exp(log_std)
    # act_predict = model.output + tf.random.normal(tf.shape(model.output)) * std
    # action_fn = K.function(inputs=[x], outputs=[act_predict])
    # return action_fn, log_probs
    return policy_fn.continuous_pg(model)


def mlp(model_input, first_layer, output_size, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    """
    Creates a fully connected neural network

    Arguments:
        model_input : Keras.Input to network
        first_layer : first input into network, this can be Keras.Input or a Keras.layer
            (will be same as model_input if not using layer from another network)
        output_size : number of neurons in output layer
        hidden_sizes : ordered list of size of each hidden layer
        activation : tf or K activation function for hidden layers
        output_activation : tf or K activation function for output layer or None for linear activation

    Returns:
        model : tf.keras model
        last_hidden_layer : tf.Layer final hidden layer
    """
    x = first_layer
    for size in hidden_sizes:
        x = layers.Dense(size, activation=activation)(x)
    last_hidden_layer = x
    acts_ph = layers.Dense(output_size, activation=output_activation)(x)
    return Model(inputs=model_input, outputs=acts_ph), last_hidden_layer


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
