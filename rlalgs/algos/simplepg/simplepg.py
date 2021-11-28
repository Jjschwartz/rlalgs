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
import tensorflow.keras.optimizers as optimizers

import rlalgs.utils.utils as utils
import rlalgs.algos.simplepg.core as core
import rlalgs.utils.preprocess as preprocess
from rlalgs.algos.models import mlp_actor, cnn_actor, print_model_summary
from rlalgs.utils.logger import Logger, setup_logger_kwargs


def simplepg(env_fn, model_fn, model_kwargs=dict(), lr=1e-2, epochs=50, batch_size=2000,
             seed=0, render=False, render_last=False, logger_kwargs=dict(),
             return_fn="simple", preprocess_fn=None, obs_dim=None, save_freq=10,
             overwrite_save=True):
    """
    Simple Policy Gradient

    Arguments:
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    model_fn : function for creating the policy gradient models to use
        (see models module for more info)
    model_kwargs : any kwargs to pass into model function
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
    logger = Logger(**logger_kwargs)
    logger.save_config(locals())

    if preprocess_fn is None:
        preprocess_fn = preprocess.preprocess_obs

    print("Initializing environment")
    env = env_fn()

    if obs_dim is None:
        obs_dim = env.observation_space.shape
    num_actions = utils.get_dim_from_space(env.action_space)

    print("Initializing Replay Buffer")
    buf = core.SimpleBuffer(obs_dim, num_actions, batch_size, return_fn)

    print("Building network")
    pi_model, pi_fn = model_fn(env, **model_kwargs)

    print_model_summary({"Actor": pi_model})

    print("Setup training ops")
    train_op = optimizers.Adam(learning_rate=lr)

    @tf.function
    def policy_loss(a_pred, a_taken, a_ret):
        action_mask = tf.one_hot(tf.cast(a_taken, tf.int32), num_actions)
        log_probs = tf.reduce_sum(action_mask * tf.nn.log_softmax(a_pred), axis=1)
        return -tf.reduce_mean(log_probs * a_ret)

    @tf.function
    def get_grads(obs, a_taken, a_ret):
        with tf.GradientTape() as tape:
            a_pred = pi_model(obs)
            loss = policy_loss(a_pred, a_taken, a_ret)
        return loss, tape.gradient(loss, pi_model.trainable_variables)

    @tf.function
    def update(batch_obs, batch_acts, batch_rets):
        loss, grads = get_grads(batch_obs, batch_acts, batch_rets)
        train_op.apply_gradients(zip(grads, pi_model.trainable_variables))
        return loss

    print("Setting up model saver")
    logger.setup_tf_model_saver(pi_model, env, "pg")

    def train_one_epoch():
        interaction_start = time.time()
        get_action_time = 0

        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False

        ep_len, ep_ret = 0, 0
        batch_ep_lens, batch_ep_rets = [], []
        t = 0

        while True:
            # render first episode of each epoch
            if not finished_rendering_this_epoch and render:
                env.render()

            o = preprocess_fn(o, env)

            a_t = time.time()
            a = pi_fn(o)
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
        batch_loss = update(batch_obs, batch_acts, batch_rets)
        print(f"Train time = {time.time() - train_start:.5f}")
        print(f"Get_action time = {get_action_time / t:.5f}")

        return batch_loss.numpy(), batch_ep_rets, batch_ep_lens

    print("Starting training")
    total_training_time = 0
    total_episodes = 0
    for i in range(epochs):
        epoch_start = time.time()
        batch_loss, batch_ep_rets, batch_ep_lens = train_one_epoch()
        epoch_time = time.time() - epoch_start

        total_training_time += epoch_time
        total_episodes += len(batch_ep_lens)

        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", batch_loss)
        logger.log_tabular("avg_return", np.mean(batch_ep_rets))
        logger.log_tabular("avg_ep_lens", np.mean(batch_ep_lens))
        logger.log_tabular("epoch_time", epoch_time)
        logger.log_tabular("total_eps", total_episodes)
        logger.log_tabular("total_training_time", total_training_time)
        logger.dump_tabular()

        if (save_freq != 0 and i % save_freq == 0) or i == epochs-1:
            itr = None if overwrite_save else i
            logger.save_model(itr)

    if render_last:
        input("Press enter to view final policy in action")
        final_ret = 0
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        while not finished_rendering_this_epoch:
            env.render()
            o = preprocess_fn(o, env)
            a = pi_fn(o)
            o, r, d, _ = env.step(a)
            final_ret += r
            if d:
                finished_rendering_this_epoch = True
        print("Final return: %.3f" % (final_ret))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("-m", "--model", type=str, default="mlp")
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

    model_fn = mlp_actor
    if args.model == "cnn":
        model_fn = cnn_actor
    model_kwargs = {"hidden_sizes": args.hidden_sizes}

    print(f"\nSimple Policy Gradient using {args.returns} returns")
    exp_name = f"simplepg_{args.returns}_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = setup_logger_kwargs(exp_name, seed=args.seed)

    simplepg(lambda: gym.make(args.env), model_fn, model_kwargs=model_kwargs, epochs=args.epochs,
             lr=args.lr, batch_size=args.batch_size, seed=args.seed, render=args.render,
             render_last=args.renderlast, return_fn=args.returns, logger_kwargs=logger_kwargs)
