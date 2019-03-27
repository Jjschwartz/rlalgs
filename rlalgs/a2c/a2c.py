"""
Synchronous Advantage Actor-Critic (A2C) implementation
"""
import time
import gym as gym
import numpy as np
from mpi4py import MPI
import tensorflow as tf
import rlalgs.a2c.mpi as mpi
import rlalgs.a2c.core as core
import rlalgs.utils.utils as utils
from rlalgs.utils.logger import Logger, OBS_NAME

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, buffer_size):
        self.obs_buf = np.zeros(utils.combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = buffer_size

    def store(self, o, a, r):
        """ Store a single step in buffer """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.ptr += 1

    def finish_path(self):
        """ Calculate and store returns for finished episode trajectory """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        ep_ret = utils.reward_to_go(ep_rews)
        self.ret_buf[path_slice] = ep_ret
        self.path_start_idx = self.ptr

    def get(self):
        """ Return stored trajectories """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.ret_buf]


def a2c(env_fn, num_cpu=4, epochs=50, steps_per_epoch=10000, hidden_sizes=[64], lr=0.001,
        gamma=0.99, seed=0, exp_name=None):
    """
    Train agent on env using A2C

    """
    # 1. Set seeds - each process is seeded differently
    seed += 10000 * mpi.proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # 2. Initialize environment
    env = env_fn()
    obs_dim = utils.get_dim_from_space(env.observation_space)
    act_dim = env.action_space.shape

    # 3a. Create global network placeholders
    obs_ph = utils.placeholder_from_space(env.observation_space, obs_space=True,
                                          name=OBS_NAME)
    act_ph = utils.placeholder_from_space(env.action_space)
    ret_ph = tf.placeholder(tf.float32, shape=(None, ))

    # 3b.Create global policy and value networks
    pi, pi_logp, v = core.mlp_actor_critic(obs_ph, act_ph, env.action_space, hidden_sizes)

    # 4. Define global losses
    pi_loss = -tf.reduce_mean(pi_logp - ret_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # 5. Define multiprocessor training ops
    pi_train_op = mpi.MPIAdamOptimizer(learning_rate=lr).minimize(pi_loss)
    v_train_op = mpi.MPIAdamOptimizer(learning_rate=lr).minimize(v_loss)

    # 6. Initialize buffer
    local_steps_per_epoch = int(steps_per_epoch / num_cpu)
    buf = ReplayBuffer(obs_dim, act_dim, local_steps_per_epoch)

    # 7. Initialize logger
    output_name = "a2c_" + env.spec.id if exp_name is None else exp_name
    logger = Logger(output_fname=output_name + ".txt")

    # 8. Create tf session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 9. Sync all params across processes
    sess.run(mpi.sync_all_params())
    # END just fucking around

    def get_action(o):
        a = sess.run(pi, {obs_ph: o.reshape(1, -1)})
        return a[0]

    def update():
        batch = buf.get()
        input_dict = {obs_ph: batch[0],
                      act_ph: batch[1],
                      ret_ph: batch[2]}

        pi_l, v_l = sess.run([pi_loss, v_loss], feed_dict=input_dict)
        # policy grad step
        sess.run([pi_train_op], feed_dict=input_dict)
        sess.run([v_train_op], feed_dict=input_dict)

        return pi_l, v_l

    # 9. The training loop
    for epoch in range(epochs):

        epoch_start = time.time()

        o, r, d = utils.process_obs(env.reset(), env.observation_space), 0, False
        ep_rews, ep_steps = [], []
        ep_r, ep_t = 0, 0

        for t in range(local_steps_per_epoch):

            a = get_action(o)
            o2, r, d, _ = env.step(a)
            o2 = utils.process_obs(o2, env.observation_space)

            ep_r += r
            ep_t += 1

            if d or t == local_steps_per_epoch-1:
                r = r if d else sess.run(v, {obs_ph: o2.reshape(1, -1)})
                buf.store(o, a, r)
                buf.finish_path()
                if d:
                    # only save if episode done
                    ep_rews.append(ep_r)
                    ep_steps.append(ep_t)
                ep_r, ep_t = 0, 0
                o, r, d = utils.process_obs(env.reset(), env.observation_space), 0, d
            else:
                buf.store(o, a, r)

            o = o2

        epoch_pi_loss, epoch_v_loss = update()

        epoch_time = time.time() - epoch_start
        if mpi.proc_id() == 0:
            logger.log_tabular("epoch", epoch)
            logger.log_tabular("pi_loss", epoch_pi_loss)
            logger.log_tabular("v_loss", epoch_v_loss)
            logger.log_tabular("avg_return", np.mean(ep_rews))
            logger.log_tabular("avg_ep_lens", np.mean(ep_steps))
            logger.log_tabular("epoch_time", epoch_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=10000)
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    # 1. fork
    mpi.mpi_fork(args.cpu)

    # 2. test fork
    mpi.print_msg("Hello, World!")

    a2c(lambda: gym.make(args.env), num_cpu=args.cpu, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, hidden_sizes=[args.hid]*args.layers,
        lr=args.lr, gamma=args.gamma, seed=args.seed, exp_name=args.exp_name)
