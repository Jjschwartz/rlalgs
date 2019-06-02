"""
A module for running multiple training runs with different seeds in parallel.

Note: each run/seed is run using a single CPU so cannot handle multiple runs of parallel
algorithms (e.g. a2c).

Functionallity:
- Inputs:
    1. An YAML file defining what's to be run, including:
        1. name of algorithm
        2. algorith arguments and hyperparams, including environment
    2. number of cpus (this will control number of simoultaneuos runs)
        - the cpu rank will also be
    3. experiment name
- Outputs:
    - results of running algorithm using algorithm arguments with different seeds the specified
    number of runs using the specified number of cpus
"""
import gym
import yaml
import rlalgs   # noqa
import rlalgs.utils.mpi as mpi
import rlalgs.utils.logger as log
from rlalgs.algos import VALID_ALGOS
import rlalgs.utils.preprocess as preprocess


def print_msg(msg):
    print(f"MultiRun {mpi.proc_id()}: {msg}")


def run_exp(exp_file, exp_name, seed):
    """
    Loads and runs the experiment
    """
    alg_name, env, args = load_exp(exp_file)
    exp_name = f"{alg_name}_{env}" if exp_name is None else exp_name
    alg_fn = eval("rlalgs."+alg_name)
    seed += mpi.proc_id()
    verbose = mpi.proc_id() == 0

    print_msg(f"Starting training")
    run_alg(alg_fn, env, args, seed, exp_name, verbose)
    print_msg(f"Finished training")


def load_exp(exp_file):
    """
    Loads an experiment from a experiment file
    """
    with open(exp_file) as fin:
        exp = yaml.load(fin, Loader=yaml.FullLoader)

    assert exp["algo"] in VALID_ALGOS, f"exp_file algo can only be one of {VALID_ALGOS}"
    algo = exp["algo"]
    env = exp["env"]
    args = exp.get("args")
    return algo, env, args


def run_alg(alg_fn, env, alg_args, seed, exp_name, verbose):
    """
    Run algorithm
    """
    logger_kwargs = log.setup_logger_kwargs(exp_name, seed=seed, verbose=verbose)
    preprocess_fn, obs_dim = preprocess.get_preprocess_fn(env)
    alg_args["seed"] = seed
    alg_args["logger_kwargs"] = logger_kwargs
    alg_args["preprocess_fn"] = preprocess_fn
    alg_args["obs_dim"] = obs_dim
    alg_fn(lambda: gym.make(env), **alg_args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_file", type=str)
    parser.add_argument("num_cpus", type=int)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    mpi.mpi_fork(args.num_cpus)

    if mpi.proc_id() == 0:
        print_msg(f"\nRunning {args.exp_name} experiment from file: {args.exp_file}")
        print_msg(f"Using {args.num_cpus} cpus")
        print_msg(f"With starting seed={args.seed}\n")

    run_exp(args.exp_file, args.exp_name, args.seed)
