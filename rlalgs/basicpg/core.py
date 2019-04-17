import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
from gym.spaces import Box, Discrete


class SimpleBuffer:
    """
    A buffer for storing trajectories (o, a, r) for simple PG without a value function
    """
    def __init__(self, finish_path_fn):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.finish_path_fn = finish_path_fn
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, o, a, r):
        """
        Store a step outcome (o, a, r) in the buffer
        """
        self.obs_buf.append(o)
        self.act_buf.append(a)
        self.rew_buf.append(r)
        self.ptr += 1

    def finish_path(self):
        """
        Called when an episode is done
        """
        ep_ret_buf = self.finish_path_fn(self.ptr, self.path_start_idx, self.rew_buf)
        self.ret_buf.extend(ep_ret_buf)
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return the stored trajectories and empty the buffer
        """
        self.ptr, self.path_start_idx = 0, 0
        obs = self.obs_buf
        acts = self.act_buf
        rets = self.ret_buf
        self.obs_buf, self.act_buf, self.rew_buf, self.ret_buf = [], [], [], []
        return obs, acts, rets

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr


def mlp(x, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    """
    Creates a fully connected neural network
    """
    for size in hidden_sizes[:-1]:
        x = tf.layers.dense(x, size, activation=activation)
    return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)


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
    act_dim = utils.get_dim_from_space(action_space)
    logits = utils.mlp(x, act_dim, hidden_sizes, activation, output_activation)
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
    act_dim = utils.get_dim_from_space(action_space)
    mu = utils.mlp(x, act_dim, hidden_sizes, activation, output_activation)
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


def actor_critic(x, a, action_space, hidden_sizes=[32], activation=tf.tanh,
                 output_activation=None):
    """
    """
    if isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    return policy(x, a, action_space, hidden_sizes, activation, output_activation)
