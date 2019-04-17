"""
Simple module that runs algorithm for a range of hyperparams and writes
results to file
"""
from rlalgs.vpg.vpg import vpg
from rlalgs.experiments.grid_tuner import GridTuner


HYPERPARAMS = {
     "pi_lr": (0.01, [0.1, 0.01, 0.001]),
     "v_lr": (0.01, [0.1, 0.01, 0.001]),
     # "gamma": (0.99, [0.9, 0.995, 1]),
     "hidden_sizes": ([64], [[32], [64], [256], [64, 64], [100, 50, 25], [400, 300]])
}


def get_hparam_arg_dicts(hparam):
    """ Create kwargs dicts for each value of hparam
        using default values of all other hparams """
    hparam_dicts = []
    for v in HYPERPARAMS[hparam][1]:
        args = {}
        args[hparam] = v
        for p, v_list in HYPERPARAMS.items():
            if p != hparam:
                args[p] = v[0]
        hparam_dicts.append(args)
    return hparam_dicts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--num_runs", type=int, default=5)
    args = parser.parse_args()

    tuner = GridTuner(name="VPG_"+args.env, seeds=args.num_runs)
    seeds = list(range(args.num_runs))
    tuner.add('env_name', args.env)
    tuner.add('epochs', 50)

    for k, v in HYPERPARAMS.items():
        tuner.add(k, v[1])

    tuner.run(vpg, num_cpu=1, data_dir="vpg_tune")
