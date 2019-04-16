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


def call_experiment(exp_name, algo, seed=0, num_cpu=1, data_dir=None, verbose=False, **kwargs):
    """
    Run an algorithm with hyperparameters (kwargs), plus configuration

    Arguments:
        exp_name : name for experiment
        algo : callable algorithm function
        seed : random number generator seed
        num_cpu : number of MPI processes to use for algo
        data_dir : directory to store experiment results. If None will use default directory.
        **kwargs : all kwargs to pass to algo

    Returns:
        results : any results returned by algorithm
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
        kwargs['logger_kwargs'] = setup_logger_kwargs(exp_name, data_dir, seed, verbose)

    if 'env_name' in kwargs:
        import gym
        env_name = kwargs['env_name']
        kwargs['env_fn'] = lambda: gym.make(env_name)
        del kwargs['env_name']

    # fork (if applicable) and run experiment
    mpi_fork(num_cpu)
    results = algo(**kwargs)

    # print end of experiment message
    logger_kwargs = kwargs['logger_kwargs']
    print("\nEnd of Experiment.\n")
    print("Results available in {}\n\n".format(logger_kwargs['output_dir']))
    print("-"*LINE_WIDTH)

    return results


class GridTuner:
    """
    Takes an algorithm and lists of hyperparam values and runs hyperparameter
    grid search.

    Three different search methods available:
    - grid search
    """

    def __init__(self, name='', seeds=[0], verbose=False):
        """
        Init an empty hyperparam tuner with given name

        Arguments:
            str name : name for the experiment. This is used when naming files
            int or list seeds : the seeds to use for runs.
                If it is a scalar this is taken to be the number of runs and so will use all seeds
                up to scalar
        """
        assert isinstance(name, str), "Name has to be string"
        assert isinstance(seeds, (list, int)), "Seeds must be a int or list of ints"
        self.name = name
        self.keys = []
        self.vals = []
        self.shs = []
        self.verbose = verbose

        if isinstance(seeds, int):
            self.seeds = list(range(seeds))
        else:
            self.seeds = seeds

    def add(self, key, vals, shorthand=None):
        """
        Add a new hyperparam with given values and optional shorthand name

        Arguments:
            key : name of the hyperparameter (must match arg name in alg function)
            vals : list of values for hyperparameter
            shorthand : optional shorthand name for hyperparam (if none, one is made from first
                three letters of key)
        """
        assert isinstance(key, str), "Key must be a string."
        assert key[0].isalnum(), "First letter of key mus be alphanumeric."
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be None or string."
        if not isinstance(vals, list):
            vals = [vals]
        if shorthand is None:
            shorthand = "".join(ch for ch in key[:3] if ch.isalnum())
        assert shorthand[0].isalnum(), "Shorthand must start with at least one alphanumeric letter."
        if key == "seed":
            print("Warning: Seeds already added to experiment so ignoring this hyperparameter addition.")
        else:
            self.keys.append(key)
            self.vals.append(vals)
            self.shs.append(shorthand)

    def print_info(self):
        """
        Prints a message containing details of tuner (i.e. current hyperparameters and their values)
        """
        print("\n", "-"*LINE_WIDTH, "\n")
        print("Hyperparameter Tuner Info:")
        table = PrettyTable()
        table.title = "HPTuner - {}".format(self.name)
        headers = ["key", "values", "shorthand"]
        table.field_names = headers
        for k, v, s in zip(self.keys, self.vals, self.shs):
            table.add_row([k, v, s])

        num_variants = int(np.prod([len(v) for v in self.vals]))

        print("\n", table, "\n")
        print("Seeds: {}".format(self.seeds))
        print("Total number of variants, ignoring seeds: {}".format(num_variants))
        print("Total number of variants, including seeds: {}\n".format(num_variants * len(self.seeds)))

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
            1. environment is also passed by user as a hyperparam

        Arguments:
            func algo : the algorothm to run (must be callable function)
            int num_cpu : number of cpus to use
            str data_dir : where the data should be output to

        Returns:
            dict results : the performance of each variant
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
        results = {}
        for var_name, var in variants:
            print("\n{} experiment {} of {}".format(self.name, exp_num, num_exps))
            print("\n" + "-"*LINE_WIDTH)
            var_result = self._run_variant(var_name, var, algo, num_cpu=num_cpu, data_dir=data_dir)
            results[var_name] = (var, var_result)
            exp_num += 1

        print("\n" + "-"*LINE_WIDTH)
        print("\nFinal results:")
        for var_name, (var, var_result) in results.items():
            print("\n\t{}:".format(var_name))
            for metric, val in var_result.items():
                print("\t\t{}: {:.3f}".format(metric, val))

        return results

    def _run_variant(self, exp_name, variant, algo, num_cpu=1, data_dir=None):
        """
        Runs a single hyperparameter setting variant with algo for each seed.
        """
        trial_num = 1
        trial_results = []
        for seed in self.seeds:
            print("\n{} Running trial {} of {}".format(">>>", trial_num, len(self.seeds)))
            print("\n" + "-"*LINE_WIDTH)
            variant["seed"] = seed
            var_result = call_experiment(exp_name, algo, num_cpu=num_cpu, data_dir=data_dir,
                                         verbose=self.verbose, **variant)
            trial_results.append(var_result)
            trial_num += 1

        results = self._analyse_trial_results(trial_results)

        print("\n" + "-"*LINE_WIDTH)
        print("\nExperiment {} complete\n".format(exp_name))
        for k, v in results.items():
            print("\t{}: {:.3f}".format(k, v))
        print("\n" + "-"*LINE_WIDTH)

        return results

    def _analyse_trial_results(self, trial_results):
        """
        Analyses trial results.

        Expects "avg_epoch_returns" to be in trial_results

        Specifically, extracts:
            1. final return = average trajectory return of last epoch
            2. average cumulative return = total average epoch return
        """
        avg_results = self._average_trial_results(trial_results)

        results = {}
        avg_epoch_returns = avg_results['avg_epoch_returns']
        results['final_return'] = avg_epoch_returns[-1]
        results['cum_return'] = np.sum(avg_epoch_returns)
        return results

    def _average_trial_results(self, trial_results):
        """
        Averages results across trials
        """
        grouped_results = {}
        for res in trial_results:
            for k, v in res.items():
                if k in grouped_results:
                    grouped_results[k].append(v)
                else:
                    grouped_results[k] = [v]

        avg_results = {}
        for k, v in grouped_results.items():
            avg_results[k] = np.mean(v, axis=0)

        return avg_results


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
