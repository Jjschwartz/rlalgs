"""
Useful functions for running tests of algorithms
"""
import yaml


BENCHMARK_FILE = "benchmarks.yaml"


def load_benchmarks():
    """
    Load benchmark yaml file

    Returns:
        dict benchmarks : dictionary with gym environment name as keys
    """
    with open(BENCHMARK_FILE, "r") as stream:
        try:
            config = yaml.load(stream)
            return config
        except yaml.YAMLError as exc:
            raise exc


def get_benchmark(env_name):
    """
    Get the benchmark score for a specific environment, if it exists

    Arguments:
        str env_name : name of gym environment

    Returns:
        int trials : number of trials to test over
        float avg_return : average return over all trials
    """
    benchmarks = load_benchmarks()
    if env_name in benchmarks:
        return benchmarks[env_name]["trials"], benchmarks[env_name]["reward"]
    else:
        return None, None


if __name__ == "__main__":
    x = load_benchmarks()
    print(x)
