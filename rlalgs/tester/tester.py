"""
Functionallity for testing implementations and trained models against some
standard benchmarks
"""
import gym
import tensorflow as tf
import rlalgs.utils.utils as utils
import rlalgs.utils.logger as logger
import rlalgs.tester.utils as testutils

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_episode(sess, env, x, pi):
    """
    Runs a single episode of the given environment for a model

    Arguments:
        sess : the tensorflow session
        env : the gym environment
        x : the policy model input tf placeholder
        pi : the policy model output tf placeholder

    Returns:
        epRew : total reward for episode
    """
    epRew = 0
    o, r, d = env.reset(), 0, False
    while not d:
        o = utils.process_obs(o, env.observation_space)
        a = sess.run(pi, {x: o.reshape(1, -1)})
        o, r, d, _ = env.step(a[0])
        epRew += r
    return epRew


def load_model(fpath):
    """
    Load a trained model from file

    Arguments:
        fpath : path to model directory

    Returns:
        sess : tensorflow sess
        x : the policy model input tf placeholder
        pi : the policy model output tf placeholder
    """
    sess = tf.Session()
    model_vars = logger.restore_model(sess, args.fpath)
    x = model_vars["inputs"][logger.OBS_NAME]
    pi = model_vars["outputs"][logger.ACTS_NAME]
    return sess, x, pi


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", metavar='fpath', type=str,
                        help="saved model directory name (i.e. the simple_save folder)")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    env_name = logger.get_env_name(args.fpath)
    trials, reward = testutils.get_benchmark(env_name)
    if trials is None or reward is None:
        print("No benchmark found for {}, please see tests.md for a list of supported envs"
              .format(env_name))
        trials = args.trials
    print("Running for {} trials".format(trials))
    env = gym.make(env_name)

    sess, x, pi = load_model(args.fpath)

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
