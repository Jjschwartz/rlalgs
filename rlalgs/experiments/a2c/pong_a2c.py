import gym
import time
import datetime
import multiprocessing
import rlalgs.utils.mpi as mpi
from rlalgs.algos.a2c.a2c import a2c
from rlalgs.utils.logger import setup_logger_kwargs
from rlalgs.utils.preprocess import preprocess_pong_image

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# cpu = 1
cpu = multiprocessing.cpu_count() - 1

mpi.mpi_fork(cpu)
mpi.print_msg(f"\nStarting Pong training using A2C and {cpu} processes")

env = "Pong-v0"
training_steps = int(16e7)
steps_per_epoch = 5000 * cpu       # > average complete episode length
epochs = int(training_steps/steps_per_epoch)
exp_name = f"a2c_{env}_{cpu}"
seed = 20

if mpi.proc_id() == 0:
    logger_kwargs = setup_logger_kwargs(exp_name, seed=seed)
else:
    logger_kwargs = dict()

params = {
    "hidden_sizes": [200],
    "epochs": epochs,
    "steps_per_epoch": steps_per_epoch,
    "pi_lr": 0.0007,        # the karpathy constant
    "vf_lr": 0.0007,
    "train_v_iters": 80,
    "gamma": 0.99,
    "seed": seed,
    "logger_kwargs": logger_kwargs,
    "save_freq": int(epochs/10),
    "overwrite_save": False,
    "preprocess_fn": preprocess_pong_image,
    "obs_dim": 80*80
}


start_time = time.time()
mpi.print_msg(f"Start time = {time.ctime()}")
a2c(lambda: gym.make(env), **params)
end_time = time.time()
mpi.print_msg(f"End time = {time.ctime()}")
total_time = end_time - start_time
mpi.print_msg(f"Total training time = {str(datetime.timedelta(seconds=total_time))}")
