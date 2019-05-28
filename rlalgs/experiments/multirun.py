"""
A module for running multiple training runs with different seeds in parallel.

Functionallity:
- Inputs:
    1. algorithms
    2. algorithm arguments, including environment
    3. seed/s
    4. number of runs
    5. number of cpus
    6. experiment name
- Outputs:
    - results of running algorithm using algorithm arguments with different seeds the specified
    number of runs using the specified number of cpus
"""
import rlalgs.utils.mpi as mpi
import rlalgs.utils.logger as log
import rlalgs.utils.preprocess as preprocess


def run_alg(alg_fn, alg_args, seed, exp_name):
    """
    Run algorithm
    """
    print("MultiRun {}: Running seed={}".format(mpi.proc_id(), seed))

    logger_kwargs = log.setup_logger_kwargs(exp_name, seed)
    alg_fn(seed=seed, logger_kwargs=logger_kwargs, **alg_args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", type=str)
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.parse_args()

    valid_algos = ["a2c", "dqn", "vpg"]
