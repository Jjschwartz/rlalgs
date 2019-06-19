"""
Implementation of Vanilla Policy Gradient Deep RL algorithm.abs

Based off of OpenAI spinning up tutorial.
"""
import gym
import time
import numpy as np
import tensorflow as tf
import rlalgs.utils.logger as log
import rlalgs.utils.utils as utils
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import rlalgs.utils.preprocess as preprocess
import tensorflow.keras.optimizers as optimizers
from rlalgs.algos.models import mlp_actor_critic
from rlalgs.algos.vpg.core import VPGReplayBuffer

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def vpg(env_fn, hidden_sizes=[64, 64], pi_lr=1e-2, v_lr=1e-2, gamma=0.99, epochs=50,
        batch_size=5000, seed=0, render=False, render_last=False, logger_kwargs=dict(),
        save_freq=10, overwrite_save=True, preprocess_fn=None, obs_dim=None):
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
    save_freq : number of epochs between model saves (always atleast saves at end of training)
    overwrite_save : whether to overwrite last saved model or save in new dir
    preprocess_fn : the preprocess function for observation. (If None then no preprocessing is
        done apart for handling reshaping for discrete observation spaces)
    obs_dim : dimensions for observations (if None then dimensions extracted from environment
        observation space)
    """
    print("Setting seeds")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("Initializing logger")
    logger = log.Logger(**logger_kwargs)
    logger.save_config(locals())

    if preprocess_fn is None:
        preprocess_fn = preprocess.preprocess_obs

    print("Initializing environment")
    env = env_fn()

    if obs_dim is None:
        obs_dim = utils.get_dim_from_space(env.observation_space)
    act_dim = env.action_space.shape

    print("Building network")
    obs_ph = layers.Input(shape=(obs_dim,))
    act_ph = utils.placeholder_from_space(env.action_space)
    ret_ph = utils.get_placeholder(tf.float32, shape=(None, ))
    adv_ph = utils.get_placeholder(tf.float32, shape=(None, ))

    pi_model, pi_fn, pi_logp, v_model, v_fn = mlp_actor_critic(
        obs_ph, act_ph, env.action_space, hidden_sizes=hidden_sizes)

    print("Setup training ops - actor")
    pi_loss = -tf.reduce_mean(pi_logp * adv_ph)
    pi_train_op = optimizers.Adam(learning_rate=pi_lr)
    pi_updates = pi_train_op.get_updates(pi_loss, pi_model.trainable_weights)
    pi_train_fn = K.function([obs_ph, act_ph, adv_ph], [pi_loss], updates=pi_updates)

    print("Setup training ops - critic")
    v_loss = tf.reduce_mean((ret_ph - v_model.output)**2)
    print("Setup training ops - 1")
    v_train_op = optimizers.Adam(learning_rate=v_lr)
    print("Setup training ops - 2")
    v_updates = v_train_op.get_updates(v_loss, v_model.trainable_weights)
    print("Setup training ops - 3")
    v_train_fn = K.function([v_model.input, ret_ph], [v_loss], updates=v_updates)
    print("Setup training ops - 4")

    print("Initializing Replay Buffer")
    buf = VPGReplayBuffer(obs_dim, act_dim, batch_size, gamma=gamma, adv_fn="gae")

    # logger.setup_tf_model_saver(sess, env, {log.OBS_NAME: obs_ph}, {log.ACTS_NAME: pi})

    def get_action(o):
        action = pi_fn([o])[0]
        return np.squeeze(action, axis=-1)

    def get_value(o):
        val = v_fn([o])[0]
        return np.squeeze(val, axis=-1)

    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        ep_len, ep_ret = 0, 0
        batch_ep_lens, batch_ep_rets = [], []
        t = 0

        while True:
            if not finished_rendering_this_epoch and render:
                env.render()

            o = preprocess_fn(o, env)
            a = get_action(o.reshape(1, -1))
            v_t = get_value(o.reshape(1, -1))
            buf.store(o, a, r, v_t)
            o, r, d, _ = env.step(a)

            ep_len += 1
            ep_ret += r
            t += 1

            if d or (t == batch_size):
                # set last_val as final reward or value of final state
                # since we may end epoch not at terminal state
                if d:
                    last_val = r
                else:
                    o = preprocess_fn(o, env)
                    last_val = get_value(o.reshape(1, -1))
                buf.finish_path(last_val)

                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
                batch_ep_lens.append(ep_len)
                batch_ep_rets.append(ep_ret)
                ep_len, ep_ret = 0, 0
                if t == batch_size:
                    break

        batch_obs, batch_acts, batch_adv, batch_rets, batch_vals = buf.get()
        pi_l = pi_train_fn([batch_obs, batch_acts, batch_adv])[0]
        v_l = v_train_fn([batch_obs, batch_rets])[0]
        return pi_l, v_l, batch_ep_rets, batch_ep_lens

    total_epoch_times = 0
    avg_epoch_returns = []
    total_episodes = 0
    for i in range(epochs):
        epoch_start = time.time()
        results = train_one_epoch()
        epoch_time = time.time() - epoch_start
        total_epoch_times += epoch_time
        avg_return = np.mean(results[2])
        total_episodes += len(results[3])
        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", results[0])
        logger.log_tabular("v_loss", results[1])
        logger.log_tabular("avg_return", avg_return)
        logger.log_tabular("avg_ep_lens", np.mean(results[3]))
        logger.log_tabular("total_eps", total_episodes)
        logger.log_tabular("epoch_time", epoch_time)
        logger.log_tabular("mem_usage", utils.get_current_mem_usage())
        avg_epoch_returns.append(avg_return)
        logger.dump_tabular()

        # if (save_freq != 0 and i % save_freq == 0) or i == epochs-1:
        #     itr = None if overwrite_save else i
        #     logger.save_model(itr)

    print("Average epoch time = ", total_epoch_times/epochs)

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
            o = preprocess_fn(o, env)
            a = get_action(o.reshape(1, -1))
            o, r, d, _ = env.step(a)
            final_ret += r
            if d:
                finished_rendering_this_epoch = True
        print("Final return: %.3f" % (final_ret))

    return {"avg_epoch_returns": avg_epoch_returns}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pi_lr", type=float, default=0.01)
    parser.add_argument("--v_lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    exp_name = "vpg_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)

    print("\nVanilla Policy Gradient")
    vpg(lambda: gym.make(args.env), epochs=args.epochs, batch_size=args.batch_size,
        hidden_sizes=[args.hid]*args.layers, pi_lr=args.pi_lr, v_lr=args.v_lr, gamma=args.gamma,
        seed=args.seed, render=args.render, render_last=args.renderlast,
        logger_kwargs=logger_kwargs, save_freq=2, overwrite_save=False)
