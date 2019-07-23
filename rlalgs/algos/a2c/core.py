"""
Core functions for Advantage Actor-Critic (A2C) implementation
"""
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
from gym.spaces import Box, Discrete


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.95, lam=0.95):
        self.obs_buf = np.zeros(utils.combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = buffer_size
        self.gamma = gamma
        self.lam = lam

    def store(self, o, a, r, v):
        """
        Store a single step in buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.ptr += 1

    def finish_path(self):
        """
        Calculate and store returns and advantage for finished episode trajectory
        Using GAE.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        # final episode step value = 0 if done, else v(st+1) = r_terminal
        final_ep_val = ep_rews[-1]
        ep_vals = np.append(self.val_buf[path_slice], final_ep_val)
        deltas = ep_rews + self.gamma * ep_vals[1:] - ep_vals[:-1]
        ep_adv = utils.discount_cumsum(deltas, self.gamma * self.lam)
        ep_ret = utils.discount_cumsum(ep_rews, self.gamma)
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_adv
        self.path_start_idx = self.ptr

    def get(self):
        """ Return stored trajectories """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.ret_buf, self.adv_buf]


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
