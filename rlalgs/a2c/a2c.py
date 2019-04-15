"""
Synchronous Advantage Actor-Critic (A2C) implementation
"""
import time
import gym as gym
import numpy as np
from mpi4py import MPI
import tensorflow as tf
import rlalgs.utils.mpi as mpi
import rlalgs.a2c.core as core
import rlalgs.utils.utils as utils
from rlalgs.utils.logger import Logger, OBS_NAME

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.95, lam=0.95):
        self.obs_buf = np.zeros(utils.combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = buffer_size
        self.gamma = gamma
        self.lam = lam

    def store(self, o, a, r, v):
        """ Store a single step in buffer """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.ptr += 1

    def finish_path(self):
        """ Calculate and store returns and advantage for finished episode trajectory
            Using GAE """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_rews = self.rew_buf[path_slice]
        # final episode step value = 0 if done, else v(st+1) = r_terminal
        final_ep_val = ep_rews[-1]
        ep_vals = np.append(self.val_buf[path_slice], final_ep_val)
        deltas = ep_rews + self.gamma * ep_vals[1:] - ep_vals[:-1]
        ep_adv = utils.discount_cumsum(deltas, self.gamma * self.lam)
        ep_ret = utils.discount_cumsum(ep_rews, self.gamma)
        self.ret_buf[path_slice] = ep_ret
        self.adv_buf[path_slice] = ep_adv
        self.path_start_idx = self.ptr

    def get(self):
        """ Return stored trajectories """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.ret_buf, self.adv_buf, self.val_buf]


def a2c(env_fn, epochs=50, steps_per_epoch=4000, hidden_sizes=[64], pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, gamma=0.99, seed=0, exp_name=None):
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
    adv_ph = tf.placeholder(tf.float32, shape=(None, ))

    # 3b.Create global policy and value networks
    pi, pi_logp, v = core.mlp_actor_critic(obs_ph, act_ph, env.action_space, hidden_sizes)

    # 4. Define global losses
    pi_loss = -tf.reduce_mean(pi_logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # 5. Define multiprocessor training ops
    pi_train_op = mpi.MPIAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    v_train_op = mpi.MPIAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # 6. Initialize buffer
    local_steps_per_epoch = int(steps_per_epoch / mpi.num_procs())
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
        a, v_t = sess.run([pi, v], {obs_ph: o.reshape(1, -1)})
        return a[0], v_t[0]

    def update():
        batch = buf.get()
        input_dict = {obs_ph: batch[0],
                      act_ph: batch[1],
                      ret_ph: batch[2],
                      adv_ph: batch[3]}

        pi_l, v_l = sess.run([pi_loss, v_loss], feed_dict=input_dict)

        # policy grad step
        sess.run([pi_train_op], feed_dict=input_dict)

        for _ in range(train_v_iters):
            # value func grad step
            sess.run([v_train_op], feed_dict=input_dict)

        return pi_l, v_l

    # 9. The training loop
    total_time = 0
    for epoch in range(epochs):

        epoch_start = time.time()

        o, r, d = utils.process_obs(env.reset(), env.observation_space), 0, False
        ep_rews, ep_steps = [], []
        ep_r, ep_t = 0, 0

        for t in range(local_steps_per_epoch):

            a, v_t = get_action(o)
            o2, r, d, _ = env.step(a)
            o2 = utils.process_obs(o2, env.observation_space)

            ep_r += r
            ep_t += 1

            if d or t == local_steps_per_epoch-1:
                r = r if d else sess.run(v, {obs_ph: o2.reshape(1, -1)})
                buf.store(o, a, r, v_t)
                buf.finish_path()
                if d:
                    # only save if episode done
                    ep_rews.append(ep_r)
                    ep_steps.append(ep_t)
                ep_r, ep_t = 0, 0
                o, r, d = utils.process_obs(env.reset(), env.observation_space), 0, d
            else:
                buf.store(o, a, r, v_t)

            o = o2

        epoch_pi_loss, epoch_v_loss = update()

        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        if mpi.proc_id() == 0:
            logger.log_tabular("epoch", epoch)
            logger.log_tabular("pi_loss", epoch_pi_loss)
            logger.log_tabular("v_loss", epoch_v_loss)
            logger.log_tabular("avg_return", np.mean(ep_rews))
            logger.log_tabular("avg_ep_lens", np.mean(ep_steps))
            logger.log_tabular("epoch_time", epoch_time)
            logger.log_tabular("time", total_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pi_lr", type=float, default=1e-3)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
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

    a2c(lambda: gym.make(args.env), epochs=args.epochs, steps_per_epoch=args.steps,
        hidden_sizes=[args.hid]*args.layers, pi_lr=args.pi_lr, vf_lr=args.vf_lr, gamma=args.gamma,
        seed=args.seed, exp_name=args.exp_name)
