"""
Some core functions for Deep Q-network implementation
"""
import numpy as np
import rlalgs.utils.utils as utils


class DQNReplayBuffer:
    """
    Replay buffer for DQN

    Store experiences (o_t, a_t, r_t, o_t+1, d_t)
    Returns a random subset of experiences for training

    Stores only the c most recent experiences, where c is the capacity of the buffer
    """

    def __init__(self, obs_dim, act_dim, capacity):
        self.obs_buf = np.zeros(utils.combined_shape(capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.obs_prime_buf = np.zeros(utils.combined_shape(capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.capacity = capacity

    def store(self, o, a, r, o_prime, d):
        """
        Store an experience (o_t, a_t, r_t, o_t+1, d_t) in the buffer
        """
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.obs_prime_buf[self.ptr] = o_prime
        self.done_buf[self.ptr] = d
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, num_samples):
        """
        Get a num_samples random samples from the replay buffer
        """
        sample_idxs = np.random.choice(self.size, num_samples)
        return {"o": self.obs_buf[sample_idxs],
                "a": self.act_buf[sample_idxs],
                "r": self.rew_buf[sample_idxs],
                "o_prime": self.obs_prime_buf[sample_idxs],
                "d": self.done_buf[sample_idxs]}
