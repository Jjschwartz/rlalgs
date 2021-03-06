"""
Replicates the original DQN papers as close as possible:
- OG paper: Mnih et al (2013)
- with target network: Mnih et al (2015)

Features of the DQN paper (for atari):
- Experience replay
    - capacity of one million most recent frames
- Used Convulutional neural net
- Minibatch size of 32
- Epsilon annealed from 1 to 0.1 over first 1 million frames
- trained for 10 million frames
"""
import gym
import time
from rlalgs import dqn
from rlalgs.utils.logger import setup_logger_kwargs
from rlalgs.utils.preprocess import preprocess_pong_image

env = "Pong-v0"
training_steps = int(5e7)   # from atari paper (50 million frames)
epoch_steps = 10000         # set to be same as target_update_freq
epochs = int(training_steps/epoch_steps)
exp_name = "dqn_pong"
seed = 50
logger_kwargs = setup_logger_kwargs(exp_name, seed=seed)

params = {
    "hidden_sizes": [400, 300],     # based of spinningup benchmark
    "lr": 0.0005,
    "epochs": epochs,
    "epoch_steps": epoch_steps,
    "batch_size": 32,   # from atari paper
    "seed": seed,
    # "replay_size": 1000000,     # from atari paper
    "replay_size": 200000,     # OG size is too big for memory, so have to use smaller size
    "epsilon": 0.1,     # from atari paper
    "gamma": 0.99,      # from atari paper
    "polyak": 0.0,     # c-step update from atari paper (i.e. not polyak updating as in spinningup)
    "start_steps": 1000000,     # from atari paper
    "target_update_freq": 10000,    # from atari paper
    "render": False,
    "render_last": False,
    "logger_kwargs": logger_kwargs,
    "save_freq": int(epochs/10),
    "overwrite_save": False,
    "preprocess_fn": preprocess_pong_image,
    "obs_dim": 80*80
}

print("\nStarting Pong training using DQN")
start_time = time.time()
print("Start time = {}\n".format(start_time))
dqn(lambda: gym.make(env), **params)
end_time = time.time()
print("\nEnd time = {}\n".format(end_time))
print("Total training time = {} hours\n".format((end_time - start_time)/3600))
