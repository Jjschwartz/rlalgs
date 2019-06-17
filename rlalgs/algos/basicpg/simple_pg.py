"""
Simplest policy gradient algorithm

Components:
- Policy network
-

Based off of OpenAI's spinningup implementation of policy gradient

N.B. some of the code is extra verbose to help with understanding
"""
import gym
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
import rlalgs.algos.basicpg.core as core
import rlalgs.utils.preprocess as preprocess

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def simple_finish_path(ptr, path_start_idx, rew_buf):
    """
    Simple PG return calculator, called when an episode is done.
    Simply sums all returns for a given episode and sets that as the
    return for each step.

    N.B. Doesn't reset the path_start_idx, this must be done outside of
    function

    Arguments:
        ptr : index of the last entry in buffers + 1
        path_start_idx : index of the start point of current episode in buffer
        rew_buf : the reward buffer

    Returns:
        ret_buf : the return buffer for the episode
    """
    path_slice = slice(path_start_idx, ptr)
    ep_len = ptr - path_start_idx
    ep_rews = rew_buf[path_slice]
    ep_ret = np.sum(ep_rews)
    ret_buf = [ep_ret] * ep_len
    return ret_buf


def simplepg(env_fn, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000,
             seed=0, render=False, render_last=False):
    """
    Simple Policy Gradient

    Arguments:
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    hidden_sizes : list of units in each hidden layer of policy network
    lr : learning rate for policy network update
    epochs : number of epochs to train for
    batch_size : max batch size for epoch
    seed : random seed
    render : whether to render environment or not
    render_last : whether to render environment after final epoch
    """

    print("Setting seeds")
    tf.set_random_seed(seed)
    np.random.seed(seed)

    print("Initializing environment")
    env = env_fn()

    print("Initializing logger")
    # logger = Logger(output_fname="simple_pg" + env.spec._env_name + ".txt")

    print("Building network")
    obs_ph = utils.placeholder_from_space(env.observation_space, obs_space=True)
    act_ph = utils.placeholder_from_space(env.action_space)
    actions, log_probs = core.actor_critic(obs_ph, act_ph, env.action_space,
                                           hidden_sizes=hidden_sizes)

    print("Setup loss")
    return_ph = tf.placeholder(tf.float32, shape=(None, ))
    loss = -tf.reduce_mean(log_probs * return_ph)

    print("Setting up training op")
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    print("Initializing Replay Buffer")
    buf = core.SimpleBuffer(simple_finish_path)

    print("Launching tf session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        # for progress reporting
        ep_len, ep_ret = 0, 0
        batch_ep_lens, batch_ep_rets = [], []

        while True:
            # render first episode of each epoch
            if (not finished_rendering_this_epoch) and render:
                env.render()
            o = preprocess.preprocess_obs(o, env)
            # select action for current obs
            a = sess.run(actions, {obs_ph: o.reshape(1, -1)})[0]
            # store step
            buf.store(o, a, r)
            # take action
            o, r, d, _ = env.step(a)
            ep_len += 1
            ep_ret += r
            # end of episode
            if d:
                buf.finish_path()
                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
                batch_ep_lens.append(ep_len)
                ep_len = 0
                batch_ep_rets.append(ep_ret)
                ep_ret = 0
                # finish epoch
                if buf.size() > batch_size:
                    break
        # get epoch trajectories
        batch_obs, batch_acts, batch_rets = buf.get()
        # take single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    return_ph: np.array(batch_rets)
                                 })
        return batch_loss, batch_ep_rets, batch_ep_lens

    print("Starting training")
    for i in range(epochs):
        batch_loss, batch_ep_rets, batch_ep_lens = train_one_epoch()
        print("\nepoch", i)
        print("loss", batch_loss)
        print("avg_return", np.mean(batch_ep_rets))
        print("avg_ep_lens", np.mean(batch_ep_lens))

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
            o = preprocess.preprocess_obs(o, env)
            a = sess.run(actions, {obs_ph: o.reshape(1, -1)})[0]
            o, r, d, _ = env.step(a)
            final_ret += r
            if d:
                finished_rendering_this_epoch = True
        print("Final return: %.3f" % (final_ret))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("\nSimple Policy Gradient")
    simplepg(lambda: gym.make(args.env), epochs=args.epochs, lr=args.lr,
             seed=args.seed, render=args.render, render_last=args.renderlast)
