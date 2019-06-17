import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
from tensorflow.keras import layers
from gym.spaces import Box, Discrete


class SimpleBuffer:
    """
    A buffer for storing trajectories (o, a, r) for simple PG without a value function
    """

    valid_fns = ["simple"]

    def __init__(self, obs_dim, act_dim, buffer_size, finish_path_fn):
        assert finish_path_fn in self.valid_fns
        self.obs_buf = np.zeros(utils.combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(buffer_size,), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.finish_path_fn = finish_path_fn
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size

        if finish_path_fn == "simple":
            self.finish_path_fn = self.simple_finish_path

    def store(self, o, a, r):
        """
        Store a step outcome (o, a, r) in the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = tf.cast(a, tf.int32)
        self.rew_buf[self.ptr] = r
        self.ptr += 1

    def finish_path(self):
        """
        Called when an episode is done
        """
        self.finish_path_fn(self.ptr, self.path_start_idx, self.rew_buf)
        self.path_start_idx = self.ptr

    def simple_finish_path(self, ptr, path_start_idx, rew_buf):
        """
        Simple PG return calculator, called when an episode is done.
        Simply sums all returns for a given episode and sets that as the
        return for each step.

        N.B. Doesn't reset the path_start_idx, this must be done outside of
        function

        Arguments:
            ptr : index of the last entry in buffers + 1
            path_start_idx : index of the start point of current episode in buffer
            rew_buf : the reward buffer

        Returns:
            ret_buf : the return buffer for the episode
        """
        path_slice = slice(path_start_idx, ptr)
        ep_len = ptr - path_start_idx
        ep_rews = rew_buf[path_slice]
        ep_ret = np.sum(ep_rews)
        ret_buf = [ep_ret] * ep_len
        self.ret_buf[path_slice] = ret_buf

    def get(self):
        """
        Return the stored trajectories and empty the buffer
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.ret_buf]

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr


class MLPCategoricalPolicy(tf.keras.Model):
    """
    A fully connected neural network for categorical output
    """
    def __init__(self,
                 num_actions,
                 hidden_sizes=[64],
                 activation=tf.tanh,
                 output_activation=None,
                 name='mlp_categorical_policy',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        # TODO: handle arbitrary number of hidden layers
        self.hidden1 = layers.Dense(hidden_sizes[0], activation=activation)
        self.logits = layers.Dense(num_actions, activation=output_activation)

    def call(self, inputs):
        """
        Get action probabilities for given inputs
        """
        # TODO: check if this is necessary
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        hidden_logs = self.hidden1(x)
        return self.logits(hidden_logs)

    def get_action(self, obs):
        """
        Returns an actions for a given observation
        """
        logits = self.predict(obs)
        action = np.squeeze(tf.random.categorical(logits, 1), axis=1)
        return action


def mlp(num_actions, hidden_sizes=[64], activation=tf.tanh, output_activation=None):
    model = tf.keras.Sequential()
    for size in hidden_sizes:
        model.add(layers.Dense(size, activation=activation))
    model.add(layers.Dense(num_actions, activation=output_activation))
    return model


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


def actor_critic(x, a, action_space, hidden_sizes=[32], activation=tf.tanh,
                 output_activation=None):
    """
    """
    if isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    return policy(x, a, action_space, hidden_sizes, activation, output_activation)
