"""
This module contains functions for creating neural network models

Model function interfaces:
- On-Policy - Policy Gradient
    - Returns:
        1. pi_model : the actor model
        2. pi_fn : the policy function (o -> a)
        3. v_model : the critic model
        4. v_fn : the value function (o -> real number value)
- Off-policy - Q Learning
    - Returns:
        1. q_network : the Q-value network
        2. pi_fn : the policy function (o -> a)
        3. q_fn : q-value fn (a, o -> max_q_val(o), q_val(a, o))
"""
import numpy as np
from gym.spaces import Box, Discrete

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import rlalgs.utils.utils as utils
import rlalgs.algos.policy as policy_fn


def print_model_summary(models):
    """Print network architecture summary for each model in models. """
    print("\nModel summaries")
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.summary()
    print()


def mlp_q_network(env, hidden_sizes=[64], activation=tf.nn.relu,
                  output_activation=None):
    """Create a fully-connected Q-network, where the output layer is the q-value for each action
    in the action space

    Arguments:
        env : the gym environment
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
    obs_dim = env.observation_space.shape
    act_dim = utils.get_dim_from_space(env.action_space)
    model_input = layers.Input(shape=obs_dim)
    q_model, _ = mlp(model_input, None, act_dim, hidden_sizes, activation, output_activation)

    pi_fn = policy_fn.discrete_qlearning(q_model)

    @tf.function
    def q_fn(a, o):
        q_raw = q_model(o)
        q_pi = tf.reduce_max(q_raw, axis=-1)
        action_mask = tf.one_hot(tf.cast(a, tf.int32), act_dim)
        act_q_val = tf.reduce_sum(action_mask * q_raw, axis=-1)
        return q_pi, act_q_val

    return q_model, pi_fn, q_fn


def cnn_q_network(env, hidden_sizes=[4, 8], activation=tf.nn.relu,
                  output_activation=None):
    """Create a CNN Q-network, where the output layer is the q-value for each action
    in the action space

    Arguments:
        env : the gym environment
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
    obs_dim = env.observation_space.shape
    act_dim = utils.get_dim_from_space(env.action_space)
    model_input = layers.Input(shape=obs_dim)
    q_model, _ = cnn(model_input, None, act_dim, hidden_sizes, activation, output_activation)

    pi_fn = policy_fn.discrete_qlearning(q_model)

    @tf.function
    def q_fn(a, o):
        q_raw = q_model(o)
        q_pi = tf.reduce_max(q_raw, axis=-1)
        action_mask = tf.one_hot(tf.cast(a, tf.int32), act_dim)
        act_q_val = tf.reduce_sum(action_mask * q_raw, axis=-1)
        return q_pi, act_q_val

    return q_model, pi_fn, q_fn


def mlp_actor_critic(env, hidden_sizes=[64], activation=tf.tanh,
                     output_activation=None, share_layers=False):
    """
    Create fully-connected policy (actor) and value (critic) networks for a continuous or
    categorical policy.

    Arguments:
        env : the gym environment
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
    obs_dim = env.observation_space.shape
    act_dim = utils.get_dim_from_space(env.action_space)
    model_input = layers.Input(shape=obs_dim)
    pi_model, last_hidden_layer = mlp(model_input, None, act_dim, hidden_sizes, activation, output_activation)

    if share_layers:
        v_model, _ = mlp(model_input, last_hidden_layer, 1, [], activation, output_activation)
    else:
        v_model, _ = mlp(model_input, None, 1, hidden_sizes, activation, output_activation)

    pi_fn = get_pg_policy(env.action_space, pi_model)
    v_fn = get_value_fn(v_model)

    return pi_model, pi_fn, v_model, v_fn


def mlp_actor(env, hidden_sizes=[32], activation=tf.tanh, output_activation=None):
    """Create a fully-connected policy (actor) neural network.

    Arguments:
        env : the gym environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        model : keras policy model
        action_fn : action selection function
        log_probs : log probabilities tensor of policy actions
    """
    obs_dim = env.observation_space.shape
    act_dim = utils.get_dim_from_space(env.action_space)
    model_input = layers.Input(shape=obs_dim)
    pi_model, _ = mlp(model_input, None, act_dim, hidden_sizes, activation, output_activation)
    pi_fn = get_pg_policy(env.action_space, pi_model)
    return pi_model, pi_fn


def cnn_actor_critic(env, hidden_sizes=[4, 8, 16], activation=tf.tanh,
                     output_activation=None, share_layers=False):
    """Create CNN policy (actor) and value (critic) networks.

    Arguments:
        env : the gym environment
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
    obs_dim = env.observation_space.shape
    act_dim = utils.get_dim_from_space(env.action_space)
    model_input = layers.Input(shape=obs_dim)
    pi_model, last_hidden_layer = cnn(model_input, None, act_dim, hidden_sizes, activation, output_activation)

    if share_layers:
        v_model, _ = cnn(model_input, last_hidden_layer, 1, [], activation, output_activation)
    else:
        v_model, _ = cnn(model_input, None, 1, hidden_sizes, activation, output_activation)

    pi_fn = get_pg_policy(env.action_space, pi_model)
    v_fn = get_value_fn(v_model)

    return pi_model, pi_fn, v_model, v_fn


def cnn_actor(env, hidden_sizes=[4, 8, 16], activation=tf.tanh, output_activation=None):
    """Create a CNN policy (actor) neural network for a continuous or categorical policy.

    Arguments:
        env : the gym environment
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        model : keras policy model
        action_fn : action selection function
        log_probs : log probabilities tensor of policy actions
    """
    obs_dim = env.observation_space.shape
    act_dim = utils.get_dim_from_space(env.action_space)
    model_input = layers.Input(shape=obs_dim)
    pi_model, _ = cnn(model_input, None, act_dim, hidden_sizes, activation, output_activation)
    pi_fn = get_pg_policy(env.action_space, pi_model)
    return pi_model, pi_fn


def get_pg_policy(action_space, model):
    """Get the policy gradient policy depending on type of actions

    Arguments:
        action_space : action space gym.space object for environment
        model : keras policy model

    Returns
        policy_fn : the policy function
    """
    if isinstance(action_space, Box):
        return policy_fn.continuous_pg(model)
    elif isinstance(action_space, Discrete):
        return policy_fn.discrete_pg(model)
    else:
        raise NotImplementedError


def get_value_fn(v_model):
    """Create function for value (critic) neural network.

    Arguments:
        v_model : keras value network

    Returns:
        v_fn : keras function for getting value for given input
    """
    @tf.function
    def model_query(o):
        return tf.squeeze(v_model(o), axis=1)

    def v_fn(o):
        v_tensor = model_query(o[np.newaxis, ...])
        return np.squeeze(v_tensor, axis=-1)
    return v_fn


def mlp(model_input, first_layer, output_size, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    """Creates a fully connected neural network

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
    if first_layer is not None:
        x = first_layer
    else:
        x = layers.Flatten()(model_input)
        for size in hidden_sizes:
            x = layers.Dense(size, activation=activation)(x)
    last_hidden_layer = x
    y = layers.Dense(output_size, activation=output_activation)(x)
    return Model(inputs=model_input, outputs=y), last_hidden_layer


def cnn(model_input, first_layer, output_size, hidden_sizes=[4, 8, 16], activation=tf.tanh,
        output_activation=None, kernel_size=3, pool_size=2):
    """Creates a convolutional neural network

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
    if first_layer is not None:
        x = first_layer
    else:
        x = model_input
        for size in hidden_sizes:
            x = layers.Conv2D(size, kernel_size, padding='same', activation=activation)(x)
            x = layers.MaxPool2D(pool_size)(x)
        x = layers.Flatten()(x)
    last_hidden_layer = x
    y = layers.Dense(output_size, activation=output_activation)(x)
    return Model(inputs=model_input, outputs=y), last_hidden_layer
