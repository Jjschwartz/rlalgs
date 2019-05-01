import gym
import time
from rlalgs.vpg.vpg import vpg
from rlalgs.utils.logger import setup_logger_kwargs

env = "Pong-v0"
training_steps = int(1e7)
batch_size = 4000
epochs = int(training_steps/batch_size)
exp_name = "vpg_pong"
seed = 40
logger_kwargs = setup_logger_kwargs(exp_name, seed=seed)

params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "hidden_sizes": [200],
    "pi_lr": 0.001,
    "v_lr": 0.001,
    "gamma": 0.99,
    "seed": seed,
    "render": False,
    "render_last": True,
    "logger_kwargs": logger_kwargs,
    "save_freq": int(epochs/10),
    "overwrite_save": False
}

print("\nStarting Pong training using VPG")
start_time = time.time()
print("Start time = {}\n".format(start_time))
vpg(lambda: gym.make(env), **params)
end_time = time.time()
print("\nEnd time = {}\n".format(end_time))
print("Total training time = {}\n".format(end_time - start_time))
