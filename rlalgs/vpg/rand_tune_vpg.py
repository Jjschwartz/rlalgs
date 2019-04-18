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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--num_exps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--metric", type=str, default="cum_return")
    args = parser.parse_args()

    tuner = RandomTuner(args.num_exps, name="VPG_"+args.env, seeds=args.num_trials, metric=args.metric)
    tuner.add('env_name', args.env)
    tuner.add('epochs', args.epochs)

    for k, v in HYPERPARAMS.items():
        if callable(v):
            tuner.add_dist(k, v)
        else:
            tuner.add(k, v)

    tuner.run(vpg, num_cpu=1, data_dir="vpg_rand_tune")
