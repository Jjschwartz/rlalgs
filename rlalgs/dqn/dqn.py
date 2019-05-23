"""
Deep Q-Network implementation using tensorflow

Replicates the original DQN paper by Mnih et al (2013) as close as possible

Features of the DQN paper (for atari):
- Experience replay
    - capacity of one million most recent frames
- Used Convulutional neural net
- Minibatch size of 32
- Epsilon annealed from 1 to 0.1 over first 1 million frames
- trained for 10 million frames
"""
import gym
import sys
import time
import numpy as np
import tensorflow as tf
import rlalgs.dqn.core as core
from gym.spaces import Discrete
import rlalgs.utils.logger as log
import rlalgs.utils.utils as utils
import rlalgs.utils.preprocess as preprocess

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQNReplayBuffer:
    """
    Replay buffer for DQN

    Store experiences (o_t, a_t, r_t, o_t+1, d_t)
    Returns a random subset of experiences for training

    Stores only the c most recent experiences, where c is the capacity of the buffer
    """

    def __init__(self, obs_dim, act_dim, capacity):
        self.obs_buf = np.zeros(utils.combined_shape(capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.obs_prime_buf = np.zeros(utils.combined_shape(capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.capacity = capacity

    def store(self, o, a, r, o_prime, d):
        """
        Store an experience (o_t, a_t, r_t, o_t+1, d_t) in the buffer
        """
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.obs_prime_buf[self.ptr] = o_prime
        self.done_buf[self.ptr] = d
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, num_samples):
        """
        Get a num_samples random samples from the replay buffer
        """
        sample_idxs = np.random.choice(self.size, num_samples)
        return {"o": self.obs_buf[sample_idxs],
                "a": self.act_buf[sample_idxs],
                "r": self.rew_buf[sample_idxs],
                "o_prime": self.obs_prime_buf[sample_idxs],
                "d": self.done_buf[sample_idxs]}


def dqn(env_fn, hidden_sizes=[64, 64], lr=1e-3, epochs=50, epoch_steps=10000, batch_size=32,
        seed=0, replay_size=100000, epsilon=0.05, gamma=0.99, polyak=0.995, start_steps=100000,
        target_update_freq=10000, render=False, render_last=False, logger_kwargs=dict(), save_freq=10,
        overwrite_save=True, preprocess_fn=None, obs_dim=None):
    """
    Deep Q-network with experience replay

    Arguments:
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    hidden_sizes : list of units in each hidden layer of policy network
    lr : learning rate for policy network update
    epochs : number of epochs to train for
    epoch_steps : number of steps per epoch
    batch_size : number of steps between main network updates
    seed : random seed
    replay_size : max size of replay buffer
    epsilon : random action selection parameter
    gamma : discount parameter
    polyak : Interpolation factor when copying target network towards main network.
    start_steps : the epsilon annealing period in number of steps
    target_update_freq : number of steps between target network updates
    render : whether to render environment or not
    render_last : whether to render environment after final epoch
    logger_kwargs : dictionary of keyword arguments for logger
    save_freq : number of epochs between model saves (always atleast saves at end of training)
    overwrite_save : whether to overwrite last saved model or save in new dir
    preprocess_fn : the preprocess function for observation. (If None then no preprocessing is
        done apart for handling reshaping for discrete observation spaces)
    obs_dim : dimensions for observations (if None then dimensions extracted from environment
        observation space)
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    if not isinstance(env.action_space, Discrete):
        raise NotImplementedError("DQN only works for environments with Discrete action spaces")

    logger = log.Logger(**logger_kwargs)
    logger.save_config(locals())

    if preprocess_fn is None:
        preprocess_fn = preprocess.preprocess_obs

    if obs_dim is None:
        obs_dim = utils.get_dim_from_space(env.observation_space)
        obs_ph = utils.placeholder_from_space(env.observation_space, obs_space=True)
        obs_prime_ph = utils.placeholder_from_space(env.observation_space, obs_space=True)
    else:
        obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
        obs_prime_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))

    # need .shape for replay buffer and #actions for random action sampling
    act_dim = env.action_space.shape
    num_actions = utils.get_dim_from_space(env.action_space)
    act_ph = utils.placeholder_from_space(env.action_space)
    rew_ph = tf.placeholder(tf.float32, shape=(None, ))
    done_ph = tf.placeholder(tf.float32, shape=(None, ))

    with tf.variable_scope("main"):
        pi, q_pi, act_q_val, q_vals = core.q_network(obs_ph, act_ph, env.action_space, hidden_sizes)

    with tf.variable_scope("target"):
        pi_targ, q_pi_targ, _, q_vals_targ = core.q_network(obs_prime_ph, act_ph, env.action_space,
                                                            hidden_sizes)

    # Losses
    target = rew_ph + gamma*(1-done_ph)*q_pi_targ
    q_loss = tf.reduce_mean((tf.stop_gradient(target) - act_q_val)**2)

    # Training ops
    q_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    q_train_op = q_optimizer.minimize(q_loss)

    # update target network to match main network
    target_init = tf.group([v_targ.assign(v_main) for v_main, v_targ
                            in zip(core.get_vars("main"), core.get_vars("target"))])

    target_update = tf.group([v_targ.assign(polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ
                              in zip(core.get_vars('main'), core.get_vars('target'))])

    buf = DQNReplayBuffer(obs_dim, act_dim, replay_size)

    epsilon_schedule = np.linspace(1, epsilon, start_steps)
    global total_t
    total_t = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    logger.setup_tf_model_saver(sess, env, {log.OBS_NAME: obs_ph}, {log.ACTS_NAME: pi})

    num_debug_states = 4
    debug_states = []

    def get_action(o):
        global total_t
        eps = epsilon if total_t >= start_steps else epsilon_schedule[total_t]
        total_t += 1
        if np.random.rand(1) < eps:
            a = np.random.choice(num_actions)
        else:
            a = sess.run(pi, {obs_ph: o.reshape(1, -1)})
        return a

    def update(t):
        batch = buf.sample(batch_size)
        feed_dict = {obs_ph: batch['o'],
                     act_ph: batch["a"],
                     rew_ph: batch["r"],
                     obs_prime_ph: batch["o_prime"],
                     done_ph: batch["d"]}

        batch_loss, _ = sess.run([q_loss, q_train_op], feed_dict)

        if t > 0 and t % target_update_freq == 0:
            sess.run(target_update)

        return batch_loss

    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        ep_len, ep_ret, ep_loss = 0, 0, []
        epoch_ep_lens, epoch_ep_rets, epoch_ep_loss = [], [], []
        t = 0

        o = preprocess_fn(o, env)
        while True:
            if not finished_rendering_this_epoch and render:
                env.render()

            if len(debug_states) < num_debug_states:
                if np.random.rand(1) < 0.1:
                    debug_states.append(o)

            a = get_action(o)
            o_prime, r, d, _ = env.step(a)
            o_prime = preprocess_fn(o_prime, env)
            buf.store(o, a, r, o_prime, d)

            batch_loss = update(t)
            ep_len += 1
            ep_ret += r
            t += 1
            ep_loss.append(batch_loss)
            o = o_prime

            if d:
                finished_rendering_this_epoch = True
                o, r, d = env.reset(), 0, False
                o = preprocess_fn(o, env)

                epoch_ep_lens.append(ep_len)
                epoch_ep_rets.append(ep_ret)
                epoch_ep_loss.append(np.mean(ep_loss))
                ep_len, ep_ret, ep_loss = 0, 0, []

            if t >= epoch_steps:
                break

        return epoch_ep_loss, epoch_ep_rets, epoch_ep_lens

    total_epoch_times = 0
    total_episodes = 0
    for i in range(epochs):
        epoch_start = time.time()
        results = train_one_epoch()
        epoch_time = time.time() - epoch_start
        total_epoch_times += epoch_time
        total_episodes += len(results[2])
        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", np.mean(results[0]))
        logger.log_tabular("avg_return", np.mean(results[1]))
        logger.log_tabular("avg_ep_lens", np.mean(results[2]))
        logger.log_tabular("total_eps", total_episodes)
        logger.log_tabular("total_steps", np.sum(results[2]))
        logger.log_tabular("end_epsilon", epsilon if total_t >= start_steps else epsilon_schedule[total_t])
        logger.log_tabular("epoch_time", epoch_time)

        for j, ds in enumerate(debug_states):
            q = sess.run([q_pi], {obs_ph: ds.reshape(1, -1)})
            logger.log_tabular("q_" + str(j), q[0])

        logger.dump_tabular()

        if (save_freq != 0 and i % save_freq == 0) or i == epochs-1:
            itr = None if overwrite_save else i
            logger.save_model(itr)

    print("Average epoch time = ", total_epoch_times/epochs)

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
            o = preprocess_fn(o, env)
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
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epoch_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--start_steps", type=int, default=100000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    exp_name = "dqn_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)

    preprocess_fn, obs_dim = preprocess.get_preprocess_fn(args.env)

    print("\nDeep Q-Network")
    dqn(lambda: gym.make(args.env), hidden_sizes=args.hidden_sizes, lr=args.lr,
        epochs=args.epochs, epoch_steps=args.epoch_steps, batch_size=args.batch_size,
        seed=args.seed, replay_size=args.replay_size, epsilon=args.epsilon, gamma=args.gamma,
        polyak=args.polyak, start_steps=args.start_steps, render=args.render,
        render_last=args.renderlast, logger_kwargs=logger_kwargs, preprocess_fn=preprocess_fn,
        obs_dim=obs_dim)
