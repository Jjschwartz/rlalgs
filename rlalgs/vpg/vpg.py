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

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class VPGReplayBuffer:
    """
    A buffer for VPG storing trajectories (o, a, r, v)
    """

    def __init__(self, use_adv=True):
        """
        Init an empty buffer

        Arguments:
            use_adv : whether to use an advantage function or not (if not then
                      naive reward-to-go is used)
        """
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.ret_buf = []
        self.adv_buf = []
        self.ptr = 0
        self.path_start_idx = 0
        self.use_adv = use_adv

    def store(self, o, a, r, v):
        """
        Store a single step outcome (o, a, r, v) in the buffer
        """
        self.obs_buf.append(o)
        self.act_buf.append(a)
        self.rew_buf.append(r)
        self.val_buf.append(v)
        self.ptr += 1

    def finish_path(self):
        """
        Called when an episode is done.
        Constructs the advantage buffer for the episode
        """
        if self.use_adv:
            self.adv_path_finish()
        else:
            self.rtg_finish_path()
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = reward_to_go(ep_rews)
        self.ret_buf.extend(ep_ret)
        self.path_start_idx = self.ptr

    def rtg_finish_path(self):
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = reward_to_go(ep_rews)
        self.ret_buf.extend(ep_ret)
        self.adv_buf.extend(ep_ret)
        self.path_start_idx = self.ptr

    def adv_path_finish(self):
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = reward_to_go(ep_rews)
        ep_vals = np.array(self.val_buf[path_slice])
        self.ret_buf.extend(ep_ret)
        self.adv_buf.extend(ep_ret - ep_vals)
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return the stored trajectories and empty the buffer
        """
        self.ptr, self.path_start_idx = 0, 0
        obs = self.obs_buf
        acts = self.act_buf
        rets = self.ret_buf
        vals = self.val_buf
        adv = self.adv_buf
        self.obs_buf, self.act_buf, self.rew_buf = [], [], []
        self.ret_buf, self.val_buf, self.adv_buf = [], [], []
        return obs, acts, adv, rets, vals

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr


def vpg(env_fn, hidden_sizes=[64], lr=1e-2, epochs=50, batch_size=5000,
        seed=0, render=False, render_last=False):
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
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    # obs_dim = utils.get_dim_from_space(env.observation_space)
    # act_dim = utils.get_dim_from_space(env.action_space)

    logger = log.Logger(output_fname="vpg_" + env.spec.id + ".txt")

    obs_ph = utils.placeholder_from_space(env.observation_space, obs_space=True,
                                          name=log.OBS_NAME)
    act_ph = utils.placeholder_from_space(env.action_space)
    ret_ph = tf.placeholder(tf.float32, shape=(None, ))
    adv_ph = tf.placeholder(tf.float32, shape=(None, ))

    pi, logp, v = mlp_actor_critic(obs_ph, act_ph, env.action_space, hidden_sizes=hidden_sizes)
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    pi_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss)
    v_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(v_loss)

    buf = VPGReplayBuffer(use_adv=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        ep_len, ep_ret = 0, 0
        batch_ep_lens, batch_ep_rets = [], []

        while True:
            if not finished_rendering_this_epoch and render:
                env.render()
            o = utils.process_obs(o, env.observation_space)
            a, v_t = sess.run([pi, v], {obs_ph: o.reshape(1, -1)})
            buf.store(o, a[0], r, v_t[0])
            o, r, d, _ = env.step(a[0])
            ep_len += 1
            ep_ret += r
            if d:
                buf.finish_path()
                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
                batch_ep_lens.append(ep_len)
                batch_ep_rets.append(ep_ret)
                ep_len, ep_ret = 0, 0
                if buf.size() > batch_size:
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

    for i in range(epochs):
        results = train_one_epoch()
        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", results[0])
        logger.log_tabular("v_loss", results[1])
        logger.log_tabular("avg_return", np.mean(results[2]))
        logger.log_tabular("avg_ep_lens", np.mean(results[3]))
        logger.dump_tabular()

    log.save_model(sess, "vpg_" + env.spec.id, env, {log.OBS_NAME: obs_ph},
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

    print("\nVanilla Policy Gradient")
    vpg(lambda: gym.make(args.env), epochs=args.epochs, lr=args.lr,
        seed=args.seed, render=args.render, render_last=args.renderlast)
