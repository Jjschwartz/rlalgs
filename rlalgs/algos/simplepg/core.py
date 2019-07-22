import numpy as np

import rlalgs.utils.utils as utils


class SimpleBuffer:
    """
    A buffer for storing trajectories (o, a, r) for simple PG without a value function
    """

    valid_fns = ["simple", "r2g"]

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
        elif finish_path_fn == "r2g":
            self.finish_path_fn = self.r2g_finish_path

    def store(self, o, a, r):
        """
        Store a step outcome (o, a, r) in the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
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
        Simply sums all returns for a given episode and sets that as the
        return for each step.
        """
        path_slice = slice(path_start_idx, ptr)
        ep_len = ptr - path_start_idx
        ep_rews = rew_buf[path_slice]
        ep_ret = np.sum(ep_rews)
        ret_buf = [ep_ret] * ep_len
        self.ret_buf[path_slice] = ret_buf

    def r2g_finish_path(self, ptr, path_start_idx, rew_buf):
        """
        The return for a given step is the sum of all rewards following that
        step in the given episode
        """
        path_slice = slice(path_start_idx, ptr)
        ep_rews = rew_buf[path_slice]
        ret_buf = self.reward_to_go(ep_rews)
        self.ret_buf[path_slice] = ret_buf

    def reward_to_go(self, rews):
        """
        Calculate the reward-to-go return for each step in a given episode
        """
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

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
