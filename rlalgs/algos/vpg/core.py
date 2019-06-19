"""
Core functions for use with Vanilla Policy Gradient (VPG) implementation
"""
import numpy as np
import rlalgs.utils.utils as utils


class VPGReplayBuffer:
    """
    A buffer for VPG storing trajectories (o, a, r, v)
    """
    valid_fns = ["simple", "adv", "gae"]

    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lmbda=0.95,
                 adv_fn="gae"):
        """
        Init an empty buffer

        Arguments:
            obs_dim : the dimensions of an environment observation
            act_dim : the dimensions of an environment action
            buffer_size : size of buffer
            gamma : gamma discount hyperparam for GAE
            lmbda : lambda hyperparam for GAE
            adv_fn : the advantage function to use, must be a value in valid_fns
                     class property
        """
        assert adv_fn in self.valid_fns
        self.obs_buf = np.zeros(utils.combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = buffer_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.adv_fn = adv_fn

    def store(self, o, a, r, v):
        """
        Store a single step outcome (o, a, r, v) in the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Called when an episode is done.
        Constructs the advantage buffer for the episode
        """
        if self.adv_fn == "simple":
            self.rtg_finish_path()
        elif self.adv_fn == "adv":
            self.adv_path_finish()
        else:
            self.gae_path_finish(last_val)
        self.path_start_idx = self.ptr

    def rtg_finish_path(self):
        """
        reward to go with no value function baseline
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = utils.reward_to_go(ep_rews)
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_ret

    def adv_path_finish(self):
        """
        Simple advantage (Q(s, a) - V(s))
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_vals = self.val_buf[path_slice]
        ep_ret = utils.reward_to_go(ep_rews)
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_ret - ep_vals

    def gae_path_finish(self, last_val=0):
        """
        General advantage estimate
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = np.append(self.rew_buf[path_slice], last_val)
        ep_vals = np.append(self.val_buf[path_slice], last_val)
        # calculate GAE
        deltas = ep_rews[:-1] + self.gamma * ep_vals[1:] - ep_vals[:-1]
        ep_adv = utils.discount_cumsum(deltas, self.gamma * self.lmbda)
        # calculate discounted reward to go for value function update
        ep_ret = utils.discount_cumsum(ep_rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_adv

    def get(self):
        """
        Return the stored trajectories and reset the buffer
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.val_buf]

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr
