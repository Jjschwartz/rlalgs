"""
Hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class
"""
import copy
import numpy as np
from prettytable import PrettyTable
from rlalgs.utils.mpi import mpi_fork
from rlalgs.utils.logger import setup_logger_kwargs


LINE_WIDTH = 60


def call_experiment(exp_name, algo, seed=0, num_cpu=1, data_dir=None, **kwargs):
    """
    Run an algorithm with hyperparameters (kwargs), plus configuration

    Arguments:
        exp_name : name for experiment
        algo : callable algorithm function
        seed : random number generator seed
        num_cpu : number of MPI processes to use for algo
        data_dir : directory to store experiment results. If None will use default directory.
        **kwargs : all kwargs to pass to algo
    """
    # in case seed not in passed kwargs dict
    kwargs['seed'] = seed

    # print experiment details
    table = PrettyTable()
    print("\nRunning experiment: {}".format(exp_name))
    table.field_names = ["Hyperparameter", "Value"]
    for k, v in kwargs.items():
        table.add_row([k, v])
    print("\n", table, "\n")

    # handle logger args and env function
    if 'logger_kwargs' not in kwargs:
        kwargs['logger_kwargs'] = setup_logger_kwargs(exp_name, data_dir, seed)

    if 'env_name' in kwargs:
        import gym
        env_name = kwargs['env_name']
        kwargs['env_fn'] = lambda: gym.make(env_name)
        del kwargs['env_name']

    # fork (if applicable) and run experiment
    mpi_fork(num_cpu)
    algo(**kwargs)

    # print end of experiment message
    logger_kwargs = kwargs['logger_kwargs']
    print("\nEnd of Experiment.\n")
    print("Results available in {}\n\n".format(logger_kwargs['output_dir']))
    print("-"*LINE_WIDTH)


class GridTuner:
    """
    Takes an algorithm and lists of hyperparam values and runs hyperparameter
    grid search.

    Three different search methods available:
    - grid search
    """

    def __init__(self, name=''):
        """
        Init an empty hyperparam tuner with given name
        """
        assert isinstance(name, str), "Name has to be string"
        self.name = name
        self.keys = []
        self.vals = []
        self.shs = []

    def add(self, key, vals, shorthand=None):
        """
        Add a new hyperparam with given values and optional shorthand name

        Arguments:
            key : name of the hyperparameter (must match arg name in alg function)
            vals : list of values for hyperparameter
            shorthand : optional shorthand name for hyperparam (if none, one is made from first
                three letters of key)
        """
        assert isinstance(key, str), "Key must be a string"
        assert key[0].isalnum(), "First letter of key mus be alphanumeric"
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be None or string."
        if not isinstance(vals, list):
            vals = [vals]
        if shorthand is None:
            shorthand = "".join(ch for ch in key[:3] if ch.isalnum())
        assert shorthand[0].isalnum(), "Shorthand must start with at least one alphanumeric letter"
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)

    def print_info(self):
        """
        Prints a message containing details of tuner (i.e. current hyperparameters and their values)
        """
        table = PrettyTable()
        table.title = "HPTuner - {}".format(self.name)
        headers = ["key", "values", "shorthand"]
        table.field_names = headers
        for k, v, s in zip(self.keys, self.vals, self.shs):
            table.add_row([k, v, s])

        num_variants = int(np.prod([len(v) for v in self.vals]))

        print("\n", table, "\n")
        print("Total number of possible variants: {} \n".format(num_variants))

    def variants(self):
        """
        Make a list of dicts, where each dict is a valid hyperparameter configuration
        """
        return self._build_variants(self.keys, self.vals)

    def _build_variants(self, keys, vals):
        """
        Recursively build hyperparameter variants
        """
        if len(keys) == 1:
            sub_variants = [dict()]
        else:
            sub_variants = self._build_variants(keys[:-1], vals[:-1])

        variants = []
        for v in vals[-1]:
            for sub_var in sub_variants:
                variant = copy.deepcopy(sub_var)
                variant[keys[-1]] = v
                variants.append(variant)
        return variants

    def name_variant(self, variant):
        """
        Get the name of variant, where the names is the HPGridTuner name followed by shorthand
        of each hyperparam and value, all seperated by underscores

        e.g.
            gridName_h1_v1_h2_v2_h3_v3 ...

        Except:
            1. does not include hyperparams with only a single value
            2. does not include seed
            3. if value is bool only include hyperparam name if val is true
        """
        var_name = self.name

        for k, v, sh in zip(self.keys, self.vals, self.shs):
            if len(v) > 1 and k != 'seed':
                variant_val = variant[k]
                if all([isinstance(val, bool) for val in v]):
                    var_name += ("_" + sh) if variant_val else ''
                else:
                    var_name += ("_" + sh + "_" + str(variant_val))
        return var_name

    def run(self, algo, num_cpu=1, data_dir=None):
        """
        Run each variant in the grid with algorithm

        Note assumes:
            1. seed is passed by user as a hyperparam in grid
                the number of seeds controls the number of runs per config
            2. environment is also passed by user as a hyperparam
        """
        self.print_info()

        variants = []
        joined_var_names = ""
        for var in self.variants():
            var_name = self.name_variant(var)
            variants.append((var_name, var))
            joined_var_names += ("\n" + var_name)

        print("-"*LINE_WIDTH)
        print("\nPreparing to run following experiments:")
        print(joined_var_names)
        print("\n" + "-"*LINE_WIDTH)

        num_exps = len(variants)
        exp_num = 1
        for var_name, var in variants:
            print("\n{} experiment {} of {}".format(self.name, exp_num, num_exps))
            print("\n" + "-"*LINE_WIDTH)
            call_experiment(var_name, algo, num_cpu=num_cpu, data_dir=data_dir, **var)
            exp_num += 1


if __name__ == "__main__":

    def dummyAlgo(seed=0, one=1, two=2, three=3, logger_kwargs=dict()):
        print("\nDummyAlgo:")
        print("\tseed:", seed)
        print("\tone:", one)
        print("\ttwo:", two)
        print("\tthree:", three)
        print("\tlogger_kwargs:", logger_kwargs)
        print("\nTraining complete. Reward = maximum awesome\n")

    tuner = GridTuner(name="Test")
    tuner.add("seed", [0, 1, 2])
    tuner.add("one", [1, 2])
    tuner.add("two", 4)
    tuner.add("three", [True, False])
    tuner.print_info()
    variants = tuner.variants()
    print("Number of variants: {}".format(len(variants)))
    for var in variants:
        print(tuner.name_variant(var), var)

    tuner.run(dummyAlgo, num_cpu=1, data_dir=None)
