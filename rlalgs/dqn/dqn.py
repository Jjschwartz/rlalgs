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
"""
TODO
- change algorithm to iterate based off of steps not episodes
- add epsilon annealing
    - change epsilon each step
    - need epsilon period
"""
import gym
import time
from gym.spaces import Discrete
import tensorflow as tf
import numpy as np
import rlalgs.utils.utils as utils
import rlalgs.dqn.core as core
import rlalgs.utils.logger as log


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


def dqn(env_fn, hidden_sizes=[64], lr=1e-2, epochs=50, epoch_episodes=100, batch_size=32,
        seed=0, replay_size=10000, epsilon=0.05, gamma=0.99, render=False, render_last=False,
        exp_name=None):
    """
    Deep Q-network with experience replay

    Arguments:
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    hidden_sizes : list of units in each hidden layer of policy network
    lr : learning rate for policy network update
    epochs : number of epochs to train for
    buffer_size : max size of replay buffer
    seed : random seed
    epsilon : random action selection parameter
    gamma : discount parameter
    render : whether to render environment or not
    render_last : whether to render environment after final epoch
    exp_name : name for experiment output files (if None, defaults to "dqn_envname")
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    if not isinstance(env.action_space, Discrete):
        raise NotImplementedError("Deep Q-network only works for environments with Discrete " +
                                  "action spaces")
    obs_dim = utils.get_dim_from_space(env.observation_space)
    # need .shape for replay buffer and #actions for random action sampling
    act_dim = env.action_space.shape
    num_actions = utils.get_dim_from_space(env.action_space)

    output_name = "dqn_" + env.spec.id if exp_name is None else exp_name
    logger = log.Logger(output_fname=output_name + ".txt")

    obs_ph = utils.placeholder_from_space(env.observation_space, obs_space=True,
                                          name=log.OBS_NAME)
    act_ph = utils.placeholder_from_space(env.action_space)
    rew_ph = tf.placeholder(tf.float32, shape=(None, ))
    obs_prime_ph = utils.placeholder_from_space(env.observation_space, obs_space=True)
    done_ph = tf.placeholder(tf.float32, shape=(None, ))

    pi, q_val_max, act_q_val = core.q_network(obs_ph, act_ph, env.action_space, hidden_sizes)
    target = rew_ph + gamma*(1-done_ph)*q_val_max

    q_loss = tf.reduce_mean((target - act_q_val)**2)
    q_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(q_loss)

    buf = DQNReplayBuffer(obs_dim, act_dim, replay_size)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_action(o):
        if np.random.rand(1) < epsilon:
            a = np.random.choice(num_actions)
        else:
            o_processed = o = utils.process_obs(o, env.observation_space)
            a = sess.run(pi, {obs_ph: o_processed.reshape(1, -1)})[0]
        return a

    def update():
        batch = buf.sample(batch_size)
        feed_dict = {obs_ph: batch['o'],
                     act_ph: batch["a"],
                     rew_ph: batch["r"],
                     obs_prime_ph: batch["o_prime"],
                     done_ph: batch["d"]
                     }
        batch_loss, _ = sess.run([q_loss, q_train_op], feed_dict)
        return batch_loss

    def train_one_epoch():
        finished_rendering_this_epoch = False
        epoch_ep_lens, epoch_ep_rets, epoch_ep_loss = [], [], []

        for e in range(epoch_episodes):
            if not finished_rendering_this_epoch and render:
                env.render()

            o, r, d = env.reset(), 0, False
            ep_len, ep_ret = 0, 0
            ep_loss = []

            while not d:
                a = get_action(o)
                o_prime, r, d, _ = env.step(a)
                buf.store(o, a, r, o_prime, d)
                ep_len += 1
                ep_ret += r
                o = o_prime
                batch_loss = update()
                ep_loss.append(batch_loss)

            finished_rendering_this_epoch = True
            epoch_ep_lens.append(ep_len)
            epoch_ep_rets.append(ep_ret)
            epoch_ep_loss.append(np.mean(ep_loss))

        return epoch_ep_loss, epoch_ep_rets, epoch_ep_lens

    for i in range(epochs):
        epoch_start = time.time()
        results = train_one_epoch()
        epoch_time = time.time() - epoch_start
        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", np.mean(results[0]))
        logger.log_tabular("avg_return", np.mean(results[1]))
        logger.log_tabular("avg_ep_lens", np.mean(results[2]))
        logger.log_tabular("epoch_time", epoch_time)
        logger.dump_tabular()

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
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    print("\nDeep Q-Network")
    dqn(lambda: gym.make(args.env), epochs=args.epochs, lr=args.lr,
        seed=args.seed, render=args.render, render_last=args.renderlast,
        exp_name=args.exp_name)