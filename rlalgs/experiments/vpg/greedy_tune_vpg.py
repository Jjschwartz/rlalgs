"""
Greedy Search Hyperparameter tuning for VPG
"""
from rlalgs.algos.vpg.vpg import vpg
from rlalgs.tuner.greedy_tuner import GreedyTuner


HYPERPARAMS = {
    "pi_lr": (0.01, [0.1, 0.01, 0.001]),
    "v_lr": (0.01, [0.1, 0.01, 0.001]),
    "hidden_sizes": ([64, 64], [[32], [64], [256], [64, 64], [100, 50, 25], [400, 300]])
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--metric", type=str, default="cum_return")
    args = parser.parse_args()

    tuner = GreedyTuner(name="VPG_"+args.env, seeds=args.num_trials, metric=args.metric)
    tuner.add('env_name', args.env)
    tuner.add('epochs', args.epochs)

    for k, v in HYPERPARAMS.items():
        tuner.add(k, v[1], default=v[0])

    tuner.run(vpg, num_cpu=1, data_dir="vpg_greedy_tune")
