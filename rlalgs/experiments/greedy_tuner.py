"""
Greedy Search based Hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class
"""
from rlalgs.experiments.tuner import Tuner


class GreedyTuner(Tuner):
    """
    Takes an algorithm and lists of hyperparam values and runs greedy hyperparameter
    search.

    Greedy search optimizes one hyperparameter at a time while using the best found or default value
    for all others.

    The order of which hyperparameter is optimized first is based on the order in which they are
    added to the tuner.

    The number of experiments run by the tuner will be equal to the total number of different
    hyperparameter settings.
    """

    def __init__(self, name='', seeds=[0], verbose=False, metric="cum_return"):
        """
        Initialize an empty greedy search hyperparameter tuner with given name
        """
        super().__init__(name, seeds, verbose, metric)
        self.best_vals = []

    def _run(self, algo, num_cpu=1, data_dir=None):
        """
        Run each variant in the grid with algorithm
        """
        results = {}
        num_exps = sum([len(v) for v in self.vals if len(v) > 0])
        exp_num = 1
        for idx in range(len(self.keys)):
            variants = self.get_hparam_variants(idx)

            # handle single value variants case
            if len(variants) == 1:
                self.best_vals.append(self.default_vals[idx])
                continue

            print("\nOptimizing hyperparameter: {}\n\n{}".format(self.keys[idx], self.thick_line))
            var_results = []
            for var in variants:
                print("{}\n{} experiment {} of {}".format(self.thick_line, self.name, exp_num, num_exps))
                var_name = self.name_variant(var)

                # only run variant if it has not already been run
                if var_name not in results:
                    var_result = self._run_variant(var_name, var, algo, num_cpu=num_cpu, data_dir=data_dir)
                    results[var_name] = var_result
                else:
                    print(self.line)
                    print("\tExperiment variant already run, so skipping.")
                    print(self.line)

                var_results.append(results[var_name])
                exp_num += 1

                print("{} experiment complete".format(var_name))
                print("\nExperiment Results:")
                self.print_results(var_results)
                print(self.thick_line)

            best_val, best_result = self.get_best_val(idx, var_results)
            self.best_vals.append(best_val)

            print("{}\n\nFinished optimizing hyperparameter: {}".format(self.thick_line, self.keys[idx]))
            print("\n\tBest value: {}".format(best_val))
            print("\n\tBest {}: {:.3f}".format(self.metric, best_result))
            print("\n" + self.thick_line)

        return list(results.values())

    def get_num_exps(self):
        return sum([len(v) for v in self.vals if len(v) > 0])

    def get_hparam_variants(self, idx):
        """
        Get the list of variants for a single hyperparam, using the best found values, where
        possible, or default values for other hyperparameters.

        Arguments:
            int idx : the index of the hyperparam to optimize

        Returns:
            list variants : variants for specified hyperparameter
        """
        variants = []
        for val in self.vals[idx]:
            variant = {}
            for i in range(len(self.best_vals)):
                variant[self.keys[i]] = self.best_vals[i]
            variant[self.keys[idx]] = val
            for i in range(idx + 1, len(self.default_vals)):
                variant[self.keys[i]] = self.default_vals[i]
            variants.append(variant)
        return variants

    def get_best_val(self, idx, var_results):
        """
        Checks the results and finds the best value of the performance metric for given
        hyperparameter.

        Arguments:
            int idx : the index of the hyperparameter that was optimized
            list var_results : list variant results

        Returns:
            varied best_val : best hyperparameter value
            float best_result : perforance metric result for best hyperparameter value
        """
        hp_key = self.keys[idx]
        best_val = None
        best_result = None
        for result in var_results:
            hp_val = result[hp_key]
            hp_result = result[self.metric]
            if best_result is None or hp_result > best_result:
                best_val = hp_val
                best_result = hp_result
        return best_val, best_result


if __name__ == "__main__":
    tuner = GreedyTuner(name="Test", seeds=5)
    tuner.add("one", [1, 2])
    tuner.add("two", [0.01, 0.0004])
    tuner.add("three", [True, False])
    tuner.print_info()

    for i in range(3):
        variants = tuner.get_hparam_variants(i)
        for var in variants:
            print(i, ":", var)
