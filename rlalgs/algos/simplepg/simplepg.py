"""
Simplest policy gradient algorithm.

Only has a policy network, and using total discounted reward as return

Components:
- Policy network

Based off of OpenAI's spinningup implementation of policy gradient

N.B. some of the code is extra verbose to help with understanding
"""
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

import rlalgs.utils.utils as utils
import rlalgs.algos.simplepg.core as core
from rlalgs.algos.models import mlp_actor
import rlalgs.utils.preprocess as preprocess
from rlalgs.utils.logger import Logger, setup_logger_kwargs


def simplepg(env_fn, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=2000,
             seed=0, render=False, render_last=False, logger_kwargs=dict(),
             return_fn="simple"):
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
    logger_kwargs : dictionary of keyword arguments for logger
    """
    print("Setting seeds")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("Initializing logger")
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())

    print("Initializing environment")
    env = env_fn()
    obs_dim = utils.get_dim_from_space(env.observation_space)
    num_actions = utils.get_dim_from_space(env.action_space)

    print("Initializing Replay Buffer")
    buf = core.SimpleBuffer(obs_dim, num_actions, batch_size, return_fn)

    print("Building network")
    obs_ph = layers.Input(shape=(obs_dim,))
    act_ph = utils.placeholder_from_space(env.action_space)
    ret_ph = utils.get_placeholder(tf.float32, shape=(None,))

    model, act_fn, log_probs = mlp_actor(obs_ph, act_ph, env.action_space, hidden_sizes)

    print("Setup training ops")
    loss = -tf.reduce_mean(log_probs * ret_ph)
    train_op = optimizers.Adam(learning_rate=lr)
    updates = train_op.get_updates(loss, model.trainable_weights)
    train_fn = K.function(inputs=[model.input, act_ph, ret_ph], outputs=[loss], updates=updates)

    def get_action(obs):
        action = act_fn([obs])[0]
        return np.squeeze(action, axis=-1)

    def train_one_epoch():
        interaction_start = time.time()
        get_action_time = 0

        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        # for progress reporting
        ep_len, ep_ret = 0, 0
        batch_ep_lens, batch_ep_rets = [], []
        t = 0

        while True:
            # render first episode of each epoch
            if not finished_rendering_this_epoch and render:
                env.render()

            o = preprocess.preprocess_obs(o, env)

            a_t = time.time()
            a = get_action(o.reshape(1, -1))
            get_action_time += (time.time() - a_t)

            buf.store(o, a, r)
            o, r, d, _ = env.step(a)

            ep_len += 1
            ep_ret += r
            t += 1

            # end of episode
            if d or (t == batch_size):
                buf.finish_path()
                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
                batch_ep_lens.append(ep_len)
                batch_ep_rets.append(ep_ret)
                ep_len, ep_ret = 0, 0
                if t == batch_size:
                    break

        print(f"Interaction time = {time.time() - interaction_start:.5f}")

        train_start = time.time()
        batch_obs, batch_acts, batch_rets = buf.get()
        batch_loss = train_fn([batch_obs, batch_acts, batch_rets])[0]
        print(f"Train time = {time.time() - train_start:.5f}")
        print(f"Get_action time = {get_action_time / t:.5f}")
        print(f"Get_action_time total = {get_action_time}")

        return batch_loss, batch_ep_rets, batch_ep_lens

    print("Starting training")
    for i in range(epochs):
        batch_loss, batch_ep_rets, batch_ep_lens = train_one_epoch()
        logger.log_tabular("epoch", i)
        logger.log_tabular("loss", batch_loss)
        logger.log_tabular("avg_return", np.mean(batch_ep_rets))
        logger.log_tabular("avg_ep_lens", np.mean(batch_ep_lens))
        logger.dump_tabular()

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
            o = preprocess.preprocess_obs(o, env)
            a = get_action(o.reshape(1, -1))
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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--returns", type=str, default="simple",
                        help="Return calculation to use ['simple', 'r2g']")
    args = parser.parse_args()

    print(f"\nSimple Policy Gradient using {args.returns} returns")
    exp_name = f"simplepg_{args.returns}_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = setup_logger_kwargs(exp_name, seed=args.seed)

    simplepg(lambda: gym.make(args.env), hidden_sizes=args.hidden_sizes, epochs=args.epochs,
             lr=args.lr, batch_size=args.batch_size, seed=args.seed, render=args.render,
             render_last=args.renderlast, return_fn=args.returns, logger_kwargs=logger_kwargs)
