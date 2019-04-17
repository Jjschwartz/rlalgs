"""
Random Search based Hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class
"""
import numpy as np
from prettytable import PrettyTable
from rlalgs.experiments.tuner import Tuner, LINE_WIDTH


class RandomTuner(Tuner):
    """
    Takes an algorithm and lists of hyperparam values and runs random hyperparameter
    search.

    The tuner will run the specified number of experiments, selecting a new random hyperparameter
    from the provided options each time.
    """

    def __init__(self, num_exps, name='', seeds=[0], verbose=False):
        """
        Initialize an empty random search hyperparameter tuner with given name

        Arguments:
            int num_exps : number of different experiment runs to conduct
            str name : name for the experiment. This is used when naming files
            int or list seeds : the seeds to use for runs.
                If it is a scalar this is taken to be the number of runs and so will use all seeds
                up to scalar
            bool verbose : whether to print detailed messages while training (True) or not (False)
        """
        super().__init__(name, seeds, verbose)
        self.num_exps = num_exps

    def print_info(self):
        """
        Prints a message containing details of tuner (i.e. current hyperparameters and their values)
        """
        print("\n", "-"*LINE_WIDTH, "\n")
        print("Random Search Hyperparameter Tuner Info:")
        table = PrettyTable()
        table.title = "Tuner - {}".format(self.name)
        headers = ["key", "values", "shorthand", "default"]
        table.field_names = headers
        for k, v, s, d in zip(self.keys, self.vals, self.shs, self.default_vals):
            v_print = 'dist' if callable(v) else v
            d_print = 'dist' if callable(d) else d
            table.add_row([k, v_print, s, d_print])

        print("\n", table, "\n")
        print("Seeds: {}".format(self.seeds))
        print("Total number of experiments, ignoring seeds: {}".format(self.num_exps))
        print("Total number of experiments, including seeds: {}\n".format(self.num_exps * len(self.seeds)))

    def add_dist(self, key, dist, shorthand=None, default=None):
        """
        Add a new hyperparam with a callable distribution that can be used to sample a value.

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
        assert callable(dist), "Dist must be callable. Use add method for lists of values"
        self.keys.append(key)
        self.vals.append(dist)
        self.shs.append(shorthand)
        self.default_vals.append(dist if default is None else default)

    def run(self, algo, num_cpu=1, data_dir=None):
        """
        Run each variant in the grid with algorithm
        """
        self.print_info()

        # construct all variants at start since np.random.seed is set each time algo is run
        # which messes with random sampling
        variants = []
        joined_var_names = ""
        for i in range(self.num_exps):
            var = self.sample_next_variant()
            var_name = self.name_variant(var)
            variants.append((var_name, var))
            joined_var_names += ("\n" + var_name)

        print("-"*LINE_WIDTH)
        print("\nPreparing to run following experiments:")
        print(joined_var_names)
        print("\n" + "-"*LINE_WIDTH)

        results = {}
        exp_num = 1
        for var_name, var in variants:
            print("\n{} experiment {} of {}".format(self.name, exp_num, self.num_exps))
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

    def sample_next_variant(self):
        """
        Radomly samples next variant.
        """
        variant = {}
        for k, v in zip(self.keys, self.vals):
            if callable(v):
                sampled_val = v()
            else:
                sampled_val = np.random.choice(v)
            variant[k] = sampled_val
        return variant


if __name__ == "__main__":
    num_exps = 16
    tuner = RandomTuner(num_exps, name="Test", seeds=5)
    tuner.add("one", [1, 2])
    tuner.add("two", [0.01, 0.0004])
    tuner.add("three", [True, False])
    tuner.add_dist("four", lambda: np.random.uniform(0, 1), "fr", 0.5)
    tuner.add_dist("five", lambda: 10, "fv", 3)
    tuner.print_info()

    for i in range(num_exps):
        var = tuner.sample_next_variant()
        print(tuner.name_variant(var), ":", var)
