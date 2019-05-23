import gym
import time
from rlalgs.dqn.dqn import dqn
from rlalgs.utils.logger import setup_logger_kwargs
from rlalgs.utils.preprocess import preprocess_pong_image

env = "Pong-v0"
training_steps = int(5e7)   # from atari paper (50 million frames)
epoch_steps = 10000         # atari paper goes by frames so this is kinda arbitrary
epochs = int(training_steps/epoch_steps)
exp_name = "dqn_pong"
seed = 30
logger_kwargs = setup_logger_kwargs(exp_name, seed=seed)

params = {
    "hidden_sizes": [400, 300],     # based of spinningup benchmark
    "lr": 0.0005,
    "epochs": epochs,
    "epoch_steps": epoch_steps,
    "batch_size": 32,   # from atari paper
    "seed": seed,
    "replay_size": 1000000,     # from atari paper
    "epsilon": 0.1,     # from atari paper
    "gamma": 0.99,      # from atari paper
    "polyak": 0.995,     # from ddpg spinningup implementation
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

print("\nStarting Pong training using VPG")
start_time = time.time()
print("Start time = {}\n".format(start_time))
dqn(lambda: gym.make(env), **params)
end_time = time.time()
print("\nEnd time = {}\n".format(end_time))
print("Total training time = {} hours\n".format((end_time - start_time)/3600))
