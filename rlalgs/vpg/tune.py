"""
Simple module that runs algorithm for a range of hyperparams and writes
results to file
"""
from rlalgs.vpg.vpg import vpg
from rlalgs.experiments.grid_tuner import GridTuner


HYPERPARAMS = {
     # "pi_lr": (0.01, [0.1, 0.01, 0.001]),
     "pi_lr": (0.01, [0.1, 0.01]),
     # "v_lr": (0.01, [0.1, 0.01, 0.001]),
     # "gamma": (0.99, [0.9, 0.99, 1]),
     # "hid_num": (1, [1, 2]),
     # "hidden_sizes": (64, [32, 64, 256])
     "hidden_sizes": ([32], [[32], [256]])
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
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()

    tuner = GridTuner(name="VPG_"+args.env)
    seeds = list(range(args.num_runs))
    tuner.add('env_name', args.env)
    tuner.add('seed', seeds)

    for k, v in HYPERPARAMS.items():
        tuner.add(k, v[1])

    tuner.print_info()

    variants = tuner.variants()
    print("Number of variants: {}".format(len(variants)))
    for var in variants:
        print(tuner.name_variant(var), var)

    tuner.run(vpg, num_cpu=1, data_dir=None)
