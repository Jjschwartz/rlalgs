"""
Module contains standard functions for preprocessing observations.
These are passed as an argument to algorithms.
"""
import numpy as np
from gym.spaces import Discrete


def preprocess_obs(o, env=None):
    """
    Standard preprocess an observation based on the observation space type
    """
    obs_space = env.observation_space
    if isinstance(obs_space, Discrete):
        return np.eye(obs_space.n)[o]
    return o


def preprocess_pong_image(o, env=None):
    """
    Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    Specific to atari pong environment.
    Credit to: http://karpathy.github.io/2016/05/31/rl/
    """
    o = o[35:195]     # crop
    o = o[::2, ::2, 0]  # downsample by factor of 2
    o[o == 144] = 0   # erase background (background type 1)
    o[o == 109] = 0   # erase background (background type 2)
    o[o != 0] = 1     # everything else (paddles, ball) just set to 1
    return o.astype(np.float).ravel()
