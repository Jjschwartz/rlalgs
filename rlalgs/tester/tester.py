"""
Functionallity for testing implementations and trained models against some
standard benchmarks
"""
import gym
import tensorflow as tf
import rlalgs.tester.utils as tutils
import rlalgs.utils.logger as logger


def run_episode(sess, env, x, pi):
    epRew = 0
    o, r, d = env.reset(), 0, False
    while not d:
        a = sess.run(pi, {x: o.reshape(1, -1)})[0]
        o, r, d, _ = env.step(a)
        epRew += r
    return epRew


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", metavar='fpath', type=str,
                        help="saved model directory name")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    env_name = logger.get_env_name(args.fpath)
    trials, reward = tutils.get_benchmark(env_name)
    if trials is None or reward is None:
        print("No benchmark found for {}, please see tests.md for a list of supported envs"
              .format(env_name))
        trials = args.trials
    print("Running for {} trials".format(trials))
    env = gym.make(env_name)

    sess = tf.Session()
    model_vars = logger.restore_model(sess, args.fpath)
    x = model_vars["inputs"][logger.OBS_NAME]
    pi = model_vars["outputs"][logger.ACTS_NAME]

    total_rew = 0
    for i in range(trials):
        ep_rew = run_episode(sess, env, x, pi)
        print("Trial {}: \t total reward = {}".format(i, ep_rew))
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
