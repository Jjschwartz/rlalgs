"""
Common general functions used by algorithm implementations
"""
from gym.spaces import Box, Discrete
import tensorflow as tf
import numpy as np


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
            return tf.placeholder(tf.float32, shape=combined_shape(None, dim), name=name)
        return tf.placeholder(tf.int32, shape=(None, ), name=name)
    elif isinstance(space, Box):
        return tf.placeholder(tf.float32, shape=combined_shape(None, space.shape), name=name)
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


def process_obs(o, obs_space):
    """
    Process an observation based on the observation space type
    """
    if isinstance(obs_space, Discrete):
        return np.eye(obs_space.n)[o]
    return o


def reward_to_go(rews):
    """
    Calculate the reward-to-go return for each step in a given episode
    """
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


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


def mlp_categorical_policy(x, a, action_space, hidden_sizes=[64], activation=tf.tanh,
                           output_activation=None):
    """
    Create a full-connected neural network for a categorical policy

    Arguments:
        x : input placeholder
        a : output placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        actions : action selection tensor
        log_probs : log probabilities tensor of policy actions
    """
    act_dim = get_dim_from_space(action_space)
    logits = mlp(x, act_dim, hidden_sizes, activation, output_activation)
    # random action selection based off raw probabilities
    actions = tf.squeeze(tf.multinomial(logits, 1), axis=1, name="pi")
    action_mask = tf.one_hot(a, act_dim)
    # Calculate the log probability for each action taken in trajectory
    # log probability = log_prob of action if action taken otherwise 0 (hence action mask)
    log_probs = action_mask * tf.nn.log_softmax(logits)
    # sum log probs for a given trajectory
    log_probs_sum = tf.reduce_sum(log_probs, axis=1)
    return actions, log_probs_sum


def mlp_gaussian_policy(x, a, action_space, hidden_sizes=[64], activation=tf.tanh,
                        output_activation=None):
    """
    Create a fully-connected neural network for a continuous policy

    Arguments:
        x : input placeholder
        a : output placeholder
        action_space : action space gym.space object for environment
        hidden_sizes : list of number of units per layer in order (including output layer)
        activation : tf activation function to use for hidden layers
        output_activation : tf activation functions to use for outq layer

    Returns:
        actions : action selection tensor
        log_probs : log probabilities tensor of policy actions
    """
    act_dim = get_dim_from_space(action_space)
    mu = mlp(x, act_dim, hidden_sizes, activation, output_activation)
    # setup log std tensor to constant value
    log_std = tf.get_variable(name="log_std", initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    actions = mu + tf.random_normal(tf.shape(mu)) * std
    log_probs = gaussian_likelihood(a, mu, log_std)
    return actions, log_probs


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
