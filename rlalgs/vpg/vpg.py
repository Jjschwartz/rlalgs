"""
Implementation of Vanilla Policy Gradient Deep RL algorithm.abs

Based off of OpenAI spinning up tutorial.
"""
import gym
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
import rlalgs.utils.logger as log
from rlalgs.vpg.core import mlp_actor_critic
import rlalgs.vpg.core as core
import time

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VPGReplayBuffer:
    """
    A buffer for VPG storing trajectories (o, a, r, v)
    """
    valid_fns = ["simple", "adv", "gae"]

    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lmbda=0.97,
                 adv_fn="simple"):
        """
        Init an empty buffer

        Arguments:
            obs_dim : the dimensions of an environment observation
            act_dim : the dimensions of an environment action
            buffer_size : size of buffer
            gamma : gamma discount hyperparam for GAE
            lmbda : lambda hyperparam for GAE
            adv_fn : the advantage function to use, must be a value in valid_fns
                     class property
        """
        assert adv_fn in self.valid_fns
        self.obs_buf = np.zeros(utils.combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = buffer_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.adv_fn = adv_fn

    def store(self, o, a, r, v):
        """
        Store a single step outcome (o, a, r, v) in the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Called when an episode is done.
        Constructs the advantage buffer for the episode
        """
        if self.adv_fn == "simple":
            self.rtg_finish_path()
        elif self.adv_fn == "adv":
            self.adv_path_finish()
        else:
            self.gae_path_finish(last_val)
        self.path_start_idx = self.ptr

    def rtg_finish_path(self):
        """
        reward to go with no value function baseline
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = utils.reward_to_go(ep_rews)
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_ret

    def adv_path_finish(self):
        """
        Simple advantage (Q(s, a) - V(s))
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_vals = self.val_buf[path_slice]
        ep_ret = utils.reward_to_go(ep_rews)
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_ret - ep_vals

    def gae_path_finish(self, last_val=0):
        """
        General advantage estimate
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = np.append(self.rew_buf[path_slice], last_val)
        ep_vals = np.append(self.val_buf[path_slice], last_val)
        # calculate GAE
        deltas = ep_rews[:-1] + self.gamma * ep_vals[1:] - ep_vals[:-1]
        ep_adv = core.discount_cumsum(deltas, self.gamma * self.lmbda)
        # calculate discounted reward to go for value function update
        ep_ret = core.discount_cumsum(ep_rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_adv

    def get(self):
        """
        Return the stored trajectories and reset the buffer
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.val_buf]

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr


def vpg(env_fn, hidden_sizes=[64], pi_lr=1e-2, v_lr=1e-3, gamma=0.99, epochs=50,
        batch_size=5000, seed=0, render=False, render_last=False, logger_kwargs=dict()):
    """
    Vanilla Policy Gradient

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
    logger_kwargs : dictionary of keyword arguments for logger
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    logger = log.Logger(**logger_kwargs)

    env = env_fn()
    obs_dim = utils.get_dim_from_space(env.observation_space)
    act_dim = env.action_space.shape

    obs_ph = utils.placeholder_from_space(env.observation_space, True, log.OBS_NAME)
    act_ph = utils.placeholder_from_space(env.action_space)
    ret_ph = tf.placeholder(tf.float32, shape=(None, ))
    adv_ph = tf.placeholder(tf.float32, shape=(None, ))

    pi, logp, v = mlp_actor_critic(obs_ph, act_ph, env.action_space, hidden_sizes=hidden_sizes)
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    pi_train_op = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    v_train_op = tf.train.AdamOptimizer(learning_rate=v_lr).minimize(v_loss)

    buf = VPGReplayBuffer(obs_dim, act_dim, batch_size, gamma=gamma, adv_fn="gae")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        ep_len, ep_ret = 0, 0
        batch_ep_lens, batch_ep_rets = [], []
        t = 0

        while True:
            if not finished_rendering_this_epoch and render:
                env.render()

            o = utils.process_obs(o, env.observation_space)

            a, v_t = sess.run([pi, v], {obs_ph: o.reshape(1, -1)})
            buf.store(o, a[0], r, v_t[0])
            o, r, d, _ = env.step(a[0])

            ep_len += 1
            ep_ret += r
            t += 1

            if d or (t == batch_size):
                # set last_val as final reward or value of final state
                # since we may end epoch not at terminal state
                if d:
                    last_val = r
                else:
                    o = utils.process_obs(o, env.observation_space)
                    last_val = sess.run(v, {obs_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)

                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
                batch_ep_lens.append(ep_len)
                batch_ep_rets.append(ep_ret)
                ep_len, ep_ret = 0, 0
                if t == batch_size:
                    break

        batch_obs, batch_acts, batch_adv, batch_rets, batch_vals = buf.get()
        inputs = {obs_ph: np.array(batch_obs),
                  act_ph: np.array(batch_acts),
                  adv_ph: np.array(batch_adv),
                  ret_ph: np.array(batch_rets),
                  }
        pi_l, v_l = sess.run([pi_loss, v_loss], feed_dict=inputs)
        sess.run(pi_train_op, feed_dict=inputs)
        sess.run(v_train_op, feed_dict=inputs)

        return pi_l, v_l, batch_ep_rets, batch_ep_lens

    total_epoch_times = 0
    for i in range(epochs):
        epoch_start = time.time()
        results = train_one_epoch()
        epoch_time = time.time() - epoch_start
        total_epoch_times += epoch_time
        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", results[0])
        logger.log_tabular("v_loss", results[1])
        logger.log_tabular("avg_return", np.mean(results[2]))
        logger.log_tabular("avg_ep_lens", np.mean(results[3]))
        logger.log_tabular("epoch_time", epoch_time)
        logger.dump_tabular()

    print("Average epoch time = ", total_epoch_times/epochs)

    log.save_model(sess, logger_kwargs["output_dir"], env, {log.OBS_NAME: obs_ph},
                   {log.ACTS_NAME: pi})

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
            a = sess.run(pi, {obs_ph: o.reshape(1, -1)})[0]
            o, r, d, _ = env.step(a)
            final_ret += r
            if d:
                finished_rendering_this_epoch = True
        print("Final return: %.3f" % (final_ret))

    return logger.get_output_filename()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--pi_lr", type=float, default=1e-2)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    exp_name = "vpg_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)

    print("\nVanilla Policy Gradient")
    vpg(lambda: gym.make(args.env), epochs=args.epochs, pi_lr=args.pi_lr, v_lr=args.v_lr,
        gamma=args.gamma, seed=args.seed, render=args.render, render_last=args.renderlast,
        logger_kwargs=logger_kwargs)
