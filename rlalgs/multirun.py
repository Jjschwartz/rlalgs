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
import yaml
import rlalgs   # noqa
import rlalgs.utils.mpi as mpi
import rlalgs.utils.logger as log
from rlalgs.algos import VALID_ALGOS
import rlalgs.utils.preprocess as preprocess


def load_exp(exp_file):
    """
    Loads an experiment from a experiment file
    """
    with open(exp_file) as fin:
        exp = yaml.load(fin, Loader=yaml.FullLoader)

    assert exp["algo"] in VALID_ALGOS, f"exp_file algo can only be one of {VALID_ALGOS}"
    algo = eval("rlalgs."+exp["algo"])

    env = exp["env"]
    



    return content



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
