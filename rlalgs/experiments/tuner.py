"""
Parent Hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class

Implementes abstract base class for hyperparameter search, which is extended by other
classes
"""
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


class Tuner:
    """
    Abstract base class for specific hyperparam search algorithms

    Subclasses must implement:
    - run
    """

    def __init__(self, name='', seeds=[0], verbose=False):
        """
        Init an empty hyperparam tuner with given name

        Arguments:
            str name : name for the experiment. This is used when naming files
            int or list seeds : the seeds to use for runs.
                If it is a scalar this is taken to be the number of runs and so will use all seeds
                up to scalar
            bool verbose : whether to print detailed messages while training (True) or not (False)
        """
        assert isinstance(name, str), "Name has to be string"
        assert isinstance(seeds, (list, int)), "Seeds must be a int or list of ints"
        self.name = name
        self.keys = []
        self.vals = []
        self.default_vals = []
        self.shs = []
        self.verbose = verbose

        if isinstance(seeds, int):
            self.seeds = list(range(seeds))
        else:
            self.seeds = seeds

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
        raise NotImplementedError

    def add(self, key, vals, shorthand=None, default=None):
        """
        Add a new hyperparam with given values and optional shorthand name

        Arguments:
            key : name of the hyperparameter (must match arg name in alg function)
            vals : list of values for hyperparameter
            shorthand : optional shorthand name for hyperparam (if none, one is made from first
                three letters of key)
            default : optional default value to use for hyperparam. If not provided, defaults to
                first value in vals list.
        """
        if key == "seed":
            print("Warning: Seeds already added to experiment so ignoring this hyperparameter addition.")
            return
        self._check_key(key)
        shorthand = self._handle_shorthand(key, shorthand)
        if not isinstance(vals, list):
            vals = [vals]
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.default_vals.append(vals[0] if default is None else default)

    def _check_key(self, key):
        """
        Checks key is valid.
        """
        assert isinstance(key, str), "Key must be a string."
        assert key[0].isalnum(), "First letter of key mus be alphanumeric."

    def _handle_shorthand(self, key, shorthand):
        """
        Handles the creation of shorthands
        """
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be None or string."
        if shorthand is None:
            shorthand = "".join(ch for ch in key[:3] if ch.isalnum())
        assert shorthand[0].isalnum(), "Shorthand must start with at least one alphanumeric letter."
        return shorthand

    def print_info(self):
        """
        Prints a message containing details of tuner (i.e. current hyperparameters and their values)
        """
        print("\n", "-"*LINE_WIDTH, "\n")
        print("Hyperparameter Tuner Info:")
        table = PrettyTable()
        table.title = "Tuner - {}".format(self.name)
        headers = ["key", "values", "shorthand", "default"]
        table.field_names = headers
        for k, v, s, d in zip(self.keys, self.vals, self.shs, self.default_vals):
            table.add_row([k, v, s, d])

        num_variants = int(np.prod([len(v) for v in self.vals]))

        print("\n", table, "\n")
        print("Seeds: {}".format(self.seeds))
        print("Total number of variants, ignoring seeds: {}".format(num_variants))
        print("Total number of variants, including seeds: {}\n".format(num_variants * len(self.seeds)))

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
            if k != 'seed' and (callable(v) or len(v) > 1):
                variant_val = variant[k]
                if not callable(v) and all([isinstance(val, bool) for val in v]):
                    var_name += ("_" + sh) if variant_val else ''
                elif callable(v):
                    if isinstance(variant_val, float):
                        val_format = "{:.3f}".format(variant_val)
                    else:
                        val_format = str(variant_val)
                    var_name += ("_" + sh + "_" + str(val_format))
                else:
                    var_name += ("_" + sh + "_" + str(variant_val))
        return var_name

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
