"""
Grid Search based hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class
"""
import copy
from rlalgs.tuner.tuner import Tuner


class GridTuner(Tuner):
    """
    Takes an algorithm and lists of hyperparam values and runs hyperparameter
    grid search.
    """

    def __init__(self, name='', seeds=[0], verbose=False, metric="mean_var"):
        """
        Init an empty grid search hyperparameter tuner with given name
        """
        super().__init__(name, seeds, verbose, metric)

    def _run(self, algo, num_cpu=1, data_dir=None):
        """
        Run each variant in the grid with algorithm
        """
        variants = []
        for var in self.variants():
            var_name = self.name_variant(var)
            variants.append((var_name, var))

        num_exps = len(variants)
        exp_num = 1
        results = []
        for var_name, var in variants:
            print("{}\n{} experiment {} of {}"
                  .format(self.thick_line, self.name, exp_num, num_exps))
            var_result = self._run_variant(var_name, var, algo, num_cpu=num_cpu, data_dir=data_dir)
            results.append(var_result)
            exp_num += 1

            print("{} experiment complete".format(var_name))
            print("\nExperiment Results:")
            self.print_results([var_result])
            print(self.thick_line)

        return results

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


if __name__ == "__main__":

    def dummyAlgo(seed=0, one=1, two=2, three=3, logger_kwargs=dict()):
        print("\nDummyAlgo:")
        print("\tseed:", seed)
        print("\tone:", one)
        print("\ttwo:", two)
        print("\tthree:", three)
        print("\tlogger_kwargs:", logger_kwargs)
        print("\nTraining complete. Reward = maximum awesome\n")

    tuner = GridTuner(name="Test", seeds=5)
    tuner.add("one", [1, 2])
    tuner.add("two", 4)
    tuner.add("three", [True, False])
    tuner.print_info()
    variants = tuner.variants()
    print("Number of variants: {}".format(len(variants)))
    for var in variants:
        print(tuner.name_variant(var), var)
