"""
Synchronous Advantage Actor-Critic (A2C) implementation
"""
import gym
import time
import numpy as np
import tensorflow as tf
import rlalgs.utils.mpi as mpi
import rlalgs.utils.logger as log
import rlalgs.utils.utils as utils
import tensorflow.keras.backend as K
import rlalgs.algos.a2c.core as core
import tensorflow.keras.layers as layers
import rlalgs.utils.preprocess as preprocess
from rlalgs.algos.models import mlp_actor_critic

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def a2c(env_fn, hidden_sizes=[64, 64], epochs=50, steps_per_epoch=4000, pi_lr=3e-4, vf_lr=1e-3,
        train_v_iters=80, gamma=0.99, seed=0, logger_kwargs=dict(), save_freq=10,
        overwrite_save=True, preprocess_fn=None, obs_dim=None):
    """
    Train agent on env using A2C

    Arguments
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    hidden_sizes : list of units in each hidden layer of policy network
    epochs : number of epochs to train for
    steps_per_epoch : number of steps per epoch
    pi_lr : learning rate for policy network update
    vf_lr : learning rate for value network update
    train_v_iters : number of value network updates per policy network update
    gamma : discount parameter
    seed : random seed
    logger_kwargs : dictionary of keyword arguments for logger
    save_freq : number of epochs between model saves (always atleast saves at end of training)
    overwrite_save : whether to overwrite last saved model or save in new dir
    preprocess_fn : the preprocess function for observation. (If None then no preprocessing is
    done apart for handling reshaping for discrete observation spaces)
    obs_dim : dimensions for observations (if None then dimensions extracted from environment
    observation space)
    """
    mpi.print_msg("Setting seeds")
    seed += 10000 * mpi.proc_id()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    mpi.print_msg("Initializing logger")
    if mpi.proc_id() == 0:
        logger = log.Logger(**logger_kwargs)
        logger.save_config(locals())

    if preprocess_fn is None:
        preprocess_fn = preprocess.preprocess_obs

    mpi.print_msg("Initializing environment")
    env = env_fn()

    if obs_dim is None:
        obs_dim = utils.get_dim_from_space(env.observation_space)
    act_dim = env.action_space.shape

    mpi.print_msg("Building network")
    obs_ph = layers.Input(shape=(obs_dim, ))
    act_ph = utils.placeholder_from_space(env.action_space)
    ret_ph = utils.get_placeholder(tf.float32, shape=(None, ))
    adv_ph = utils.get_placeholder(tf.float32, shape=(None, ))

    pi_model, pi_fn, pi_logp, v_model, v_fn = mlp_actor_critic(obs_ph, act_ph, env.action_space, hidden_sizes)

    mpi.print_msg("Setup training ops - actor")
    pi_loss = -tf.reduce_mean(pi_logp * adv_ph)
    pi_train_op = mpi.MPIAdamOptimizer(learning_rate=pi_lr)
    pi_updates = pi_train_op.get_updates(pi_loss, pi_model.trainable_weights)
    pi_train_fn = K.function([obs_ph, act_ph, adv_ph], [pi_loss], updates=pi_updates)

    mpi.print_msg("Setup training ops - critic")
    v_loss = tf.reduce_mean((ret_ph - v_model.output)**2)
    v_train_op = mpi.MPIAdamOptimizer(learning_rate=vf_lr)
    v_updates = v_train_op.get_updates(v_loss, v_model.trainable_weights)
    v_train_fn = K.function([v_model.input, ret_ph], [v_loss], updates=v_updates)

    mpi.print_msg("Initializing Replay Buffer")
    local_steps_per_epoch = int(steps_per_epoch / mpi.num_procs())
    buf = core.ReplayBuffer(obs_dim, act_dim, local_steps_per_epoch)

    # 9. Sync all params across processes
    mpi.print_msg("Syncing all params")
    mpi.sync_params([pi_model.trainable_weights, v_model.trainable_weights])

    # if mpi.proc_id() == 0:
    #     # only save model of one cpu
    #     logger.setup_tf_model_saver(sess, env, {log.OBS_NAME: obs_ph}, {log.ACTS_NAME: pi})

    def get_action(o):
        action = pi_fn([o])[0]
        return np.squeeze(action, axis=-1)

    def get_value(o):
        val = v_fn([o])[0]
        return np.squeeze(val, axis=-1)

    def update():
        batch_obs, batch_acts, batch_rets, batch_adv = buf.get()
        pi_l = pi_train_fn([batch_obs, batch_acts, batch_adv])[0]
        for _ in range(train_v_iters):
            v_l = v_train_fn([batch_obs, batch_rets])[0]
        mpi.sync_params([pi_model.trainable_weights, v_model.trainable_weights])
        return pi_l, v_l

    # 9. The training loop
    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        batch_ep_rets, batch_ep_lens = [], []
        ep_ret, ep_len = 0, 0

        for t in range(local_steps_per_epoch):

            o = preprocess_fn(o, env)
            a = get_action(o.reshape(1, -1))
            v_t = get_value(o.reshape(1, -1))
            o, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1

            if d or t == local_steps_per_epoch-1:
                if not d:
                    r = get_value(o.reshape(1, -1))
                buf.store(o, a, r, v_t)
                buf.finish_path()
                if d:
                    # only save if episode done
                    batch_ep_rets.append(ep_ret)
                    batch_ep_lens.append(ep_len)
                ep_ret, ep_len = 0, 0
                o, r, d = env.reset(), env, 0, False
            else:
                buf.store(o, a, r, v_t)

        epoch_pi_loss, epoch_v_loss = update()
        return epoch_pi_loss, epoch_v_loss, batch_ep_rets, batch_ep_lens

    total_time = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        results = train_one_epoch()
        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        if mpi.proc_id() == 0:
            logger.log_tabular("epoch", epoch)
            logger.log_tabular("pi_loss", results[0])
            logger.log_tabular("v_loss", results[1])
            logger.log_tabular("avg_return", np.mean(results[2]))
            logger.log_tabular("avg_ep_lens", np.mean(results[3]))
            logger.log_tabular("epoch_time", epoch_time)
            logger.log_tabular("time", total_time)

            training_time_left = utils.training_time_left(epoch, epochs, epoch_time)
            logger.log_tabular("time_rem", training_time_left)
            logger.dump_tabular()

            # if (save_freq != 0 and epoch % save_freq == 0) or epoch == epochs-1:
            #     itr = None if overwrite_save else epoch
            #     logger.save_model(itr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--pi_lr", type=float, default=1e-3)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--train_v_iters", type=int, default=80)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    # 1. fork
    mpi.mpi_fork(args.cpu)

    # 2. test fork
    mpi.print_msg("Running A2C!")

    exp_name = f"a2c_{args.cpu}_{args.env}" if args.exp_name is None else args.exp_name
    if mpi.proc_id() == 0:
        logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)
    else:
        logger_kwargs = dict()

    preprocess_fn, obs_dim = preprocess.get_preprocess_fn(args.env)

    a2c(lambda: gym.make(args.env), hidden_sizes=args.hidden_sizes, epochs=args.epochs,
        steps_per_epoch=args.steps, pi_lr=args.pi_lr, vf_lr=args.vf_lr, seed=args.seed,
        train_v_iters=args.train_v_iters, gamma=args.gamma, logger_kwargs=logger_kwargs,
        preprocess_fn=preprocess_fn, obs_dim=obs_dim)
