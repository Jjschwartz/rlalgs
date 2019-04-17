"""
Random Search Hyperparameter tuning for VPG
"""
import numpy as np
from rlalgs.vpg.vpg import vpg
from rlalgs.experiments.random_tuner import RandomTuner


HYPERPARAMS = {
     "pi_lr": lambda: np.random.uniform(0.001, 0.1),
     "v_lr": lambda: np.random.uniform(0.001, 0.1),
     # "gamma": (0.99, [0.9, 0.995, 1]),
     "hidden_sizes": [[32], [64], [256], [64, 64], [100, 50, 25], [400, 300]]
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
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--num_exps", type=int, default=16)
    args = parser.parse_args()

    tuner = RandomTuner(args.num_exps, name="VPG_"+args.env, seeds=args.num_trials)
    tuner.add('env_name', args.env)
    tuner.add('epochs', 50)

    for k, v in HYPERPARAMS.items():
        if callable(v):
            tuner.add_dist(k, v)
        else:
            tuner.add(k, v)

    tuner.run(vpg, num_cpu=1, data_dir="vpg_rand_tune")
