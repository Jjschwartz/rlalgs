"""
Functionallity for testing implementations and trained models against some
standard benchmarks
"""
import gym
import time

import rlalgs.utils.logger as logger
import rlalgs.tester.utils as testutils
import rlalgs.utils.preprocess as preprocess


def run_episode(env, pi_fn, preprocess_fn, render):
    """
    Runs a single episode of the given environment for a model

    Arguments:
        env : the gym environment
        pi_fn : policy function
        preprocess_fn : observation processing function
        render : whether to render episode (True) or not (False)

    Returns:
        epRew : total reward for episode
        t : total time steps for episode
    """
    epRew = 0
    o, r, d = env.reset(), 0, False
    t = 0
    a_time = 0
    while not d:
        if render:
            env.render()
            time.sleep(0.01)
        o = preprocess_fn(o, env)
        a_start = time.time()
        a = pi_fn(o)
        a_time += (time.time() - a_start)
        o, r, d, _ = env.step(a)
        epRew += r
        t += 1
    print(f"Get action time: {a_time/t:.5f} sec")
    return epRew, t


def load_policy(fpath):
    """
    Load a trained policy from file

    Arguments:
        fpath : path to model directory

    Returns:
        pi_fn : the action selection function
    """
    _, _, pi_fn = logger.restore_model(fpath)
    return pi_fn


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", metavar='fpath', type=str,
                        help="saved model directory name (i.e. the simple_save folder)")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env_name = logger.get_env_name(args.fpath)
    trials, reward = testutils.get_benchmark(env_name)
    if trials is None or reward is None:
        print("No benchmark found for {}, please see tests.md for a list of supported envs"
              .format(env_name))
        trials = args.trials
    print("Running for {} trials".format(trials))
    env = gym.make(env_name)

    pi_fn = load_policy(args.fpath)
    preprocess_fn, _ = preprocess.get_preprocess_fn(env_name)

    total_rew = 0
    for i in range(trials):
        ep_rew, t = run_episode(env, pi_fn, preprocess_fn, args.render)
        print("Trial {}: \t total reward = {}, total steps = {}".format(i, ep_rew, t))
        total_rew += ep_rew

    print("-" * 20, "\n")
    print("Test finished")
    print("Average reward over {} trials = {}".format(trials, total_rew / trials))
    if reward is not None:
        print("Benchmark reward = {}\n".format(reward))
        if total_rew / trials > reward:
            print("Benchmark passed\n")
        else:
            print("Benchmark failed\n")
    print("-" * 20, "\n")
