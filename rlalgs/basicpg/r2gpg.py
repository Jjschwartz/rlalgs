"""
Simplest reward-to-go policy gradient algorithm using

Based off of OpenAI's spinningup implementation of policy gradient

N.B. some of the code is extra verbose to help with understanding
"""
import numpy as np
import gym
import tensorflow as tf
import rlalgs.basicpg.core as core

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class ReplayBuffer:
    """
    A buffer for storing trajectories (o, a, r)
    """

    def __init__(self, obs_dim, act_dim, buffer_size):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, o, a, r):
        """
        Store a step outcome (o, a, r) in the buffer
        """
        self.obs_buf.append(o)
        self.act_buf.append(a)
        self.rew_buf.append(r)
        self.ptr += 1

    def finish_path(self):
        """
        Called when an episode is done
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = reward_to_go(ep_rews)
        self.ret_buf.extend(ep_ret)
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return the stored trajectories and empty the buffer
        """
        self.ptr, self.path_start_idx = 0, 0
        obs = self.obs_buf
        acts = self.act_buf
        rets = self.ret_buf
        self.obs_buf, self.act_buf, self.rew_buf, self.ret_buf = [], [], [], []
        return obs, acts, rets

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr


def r2gpg(env_fn, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000,
          seed=0, render=False, render_last=False):
    """
    Simple Reward-to-Go Policy Gradient

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

    # instantiate environment
    print("Initializing environment")
    env = env_fn()
    print(env.observation_space)
    print(env.observation_space.shape)
    # obs_dim = core.get_dim_from_space(env.observation_space)
    obs_dim = 1
    act_dim = core.get_dim_from_space(env.action_space)

    print(obs_dim)
    print(act_dim)

    # build policy network
    print("Building network")
    # obs_ph = core.placeholder_from_space(env.observation_space)
    obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    print(obs_ph.shape)
    act_ph = core.placeholder_from_space(env.action_space)
    actions, log_probs = core.actor_critic(obs_ph, act_ph, env.action_space,
                                           hidden_sizes=hidden_sizes)

    print("Building loss function")
    return_ph = tf.placeholder(tf.float32, shape=(None, ))  # takes batch of trajectory return
    loss = -tf.reduce_mean(log_probs * return_ph)

    print("Setting up training op")
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    print("Initializing Replay Buffer")
    buf = ReplayBuffer(obs_dim, act_dim, batch_size)

    print("Launching tf session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():

        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False

        while True:
            # render first episode of each epoch
            if (not finished_rendering_this_epoch) and render:
                env.render()

            if np.isscalar(o):
                o = np.array(o)
            # select action for current obs
            a = sess.run(actions, {obs_ph: o.reshape(1, -1)})[0]
            # store step
            buf.store(o, a, r)
            # take action
            o, r, d, _ = env.step(a)
            # end of episode
            if d:
                buf.finish_path()
                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
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
        return batch_loss, batch_rets

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets = train_one_epoch()
        print("epoch: %3d \t loss: %.3f \t return: %.3f" %
              (i, batch_loss, np.mean(batch_rets)))

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
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
    args = parser.parse_args()

    print("\nSimple Reward-to-Go Policy Gradient")
    r2gpg(lambda: gym.make(args.env), epochs=args.epochs, lr=args.lr,
          render=args.render, render_last=args.renderlast)
