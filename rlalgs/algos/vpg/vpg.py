"""
Implementation of Vanilla Policy Gradient Deep RL algorithm.abs

Based off of OpenAI spinning up tutorial.
"""
import gym
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers

import rlalgs.utils.logger as log
import rlalgs.utils.utils as utils
import rlalgs.utils.preprocess as preprocess
from rlalgs.algos.buffers import PGReplayBuffer
from rlalgs.algos.models import mlp_actor_critic, cnn_actor_critic, print_model_summary


def vpg(env_fn, model_fn, model_kwargs, pi_lr=1e-2, v_lr=1e-2, gamma=0.99, epochs=50,
        batch_size=5000, seed=0, render=False, render_last=False, logger_kwargs=dict(),
        save_freq=10, overwrite_save=True, preprocess_fn=None, obs_dim=None):
    """Vanilla Policy Gradient

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
    logger = log.Logger(**logger_kwargs)
    logger.save_config(locals())

    if preprocess_fn is None:
        preprocess_fn = preprocess.preprocess_obs

    print("Initializing environment")
    env = env_fn()

    if obs_dim is None:
        obs_dim = env.observation_space.shape
    num_actions = utils.get_dim_from_space(env.action_space)
    act_dim = env.action_space.shape

    print("Initializing Replay Buffer")
    buf = PGReplayBuffer(obs_dim, act_dim, batch_size, gamma=gamma, adv_fn="gae")

    print("Building network")
    pi_model, pi_fn, v_model, v_fn = model_fn(env, **model_kwargs)

    print_model_summary({"Actor": pi_model, "Critic": v_model})

    print("Setup training ops - actor")
    pi_train_op = optimizers.Adam(learning_rate=pi_lr)

    @tf.function
    def policy_loss(a_pred, a_taken, a_adv):
        action_mask = tf.one_hot(tf.cast(a_taken, tf.int32), num_actions)
        log_probs = tf.reduce_sum(action_mask * tf.nn.log_softmax(a_pred), axis=1)
        return -tf.reduce_mean(log_probs * a_adv)

    print("Setup training ops - critic")
    v_train_op = optimizers.Adam(learning_rate=v_lr)

    @tf.function
    def value_loss(o_val, o_ret):
        return tf.reduce_mean((o_ret - o_val)**2)

    @tf.function
    def get_grads(batch_obs, batch_acts, batch_rets, batch_adv):
        with tf.GradientTape(persistent=True) as tape:
            a_pred = pi_model(batch_obs)
            o_val = v_model(batch_obs)
            pi_loss = policy_loss(a_pred, batch_acts, batch_adv)
            v_loss = value_loss(o_val, batch_rets)
        pi_grads = tape.gradient(pi_loss, pi_model.trainable_variables)
        v_grads = tape.gradient(v_loss, v_model.trainable_variables)
        return pi_loss, pi_grads, v_loss, v_grads

    @tf.function
    def apply_gradients(pi_grads, v_grads):
        pi_train_op.apply_gradients(zip(pi_grads, pi_model.trainable_variables))
        v_train_op.apply_gradients(zip(v_grads, v_model.trainable_variables))

    @tf.function
    def update(batch_obs, batch_acts, batch_rets, batch_adv):
        pi_loss, pi_grads, v_loss, v_grads = get_grads(
            batch_obs, batch_acts, batch_rets, batch_rets)
        apply_gradients(pi_grads, v_grads)
        return pi_loss, v_loss

    print("Setting up model saver")
    logger.setup_tf_model_saver(pi_model, env, "pg", v_model)

    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        batch_ep_lens, batch_ep_rets = [], []
        ep_len, ep_ret = 0, 0
        t = 0

        while True:
            if not finished_rendering_this_epoch and render:
                env.render()

            o = preprocess_fn(o, env)
            a = pi_fn(o)
            v_t = v_fn(o)
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
                    # only save completed episodes for reporting
                    batch_ep_lens.append(ep_len)
                    batch_ep_rets.append(ep_ret)
                else:
                    o = preprocess_fn(o, env)
                    last_val = v_fn(o)
                buf.finish_path(last_val)

                o, r, d = env.reset(), 0, False
                finished_rendering_this_epoch = True
                ep_len, ep_ret = 0, 0
                if t == batch_size:
                    break

        batch_obs, batch_acts, batch_adv, batch_rets, batch_vals = buf.get()
        pi_loss, v_loss = update(batch_obs, batch_acts, batch_rets, batch_adv)
        return pi_loss.numpy(), v_loss.numpy(), batch_ep_rets, batch_ep_lens

    total_training_time = 0
    total_episodes = 0
    for i in range(epochs):
        epoch_start = time.time()
        results = train_one_epoch()

        epoch_time = time.time() - epoch_start
        total_training_time += epoch_time
        avg_return = np.mean(results[2])
        total_episodes += len(results[3])

        logger.log_tabular("epoch", i)
        logger.log_tabular("pi_loss", results[0])
        logger.log_tabular("v_loss", results[1])
        logger.log_tabular("avg_return", avg_return)
        logger.log_tabular("avg_ep_lens", np.mean(results[3]))
        logger.log_tabular("epoch_time", epoch_time)
        logger.log_tabular("total_eps", total_episodes)
        logger.log_tabular("total_time", total_training_time)
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pi_lr", type=float, default=0.01)
    parser.add_argument("--v_lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    exp_name = "vpg_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)

    model_fn = mlp_actor_critic
    if args.model == "cnn":
        model_fn = cnn_actor_critic
    model_kwargs = {"hidden_sizes": args.hidden_sizes, "share_layers": True}

    print("\nVanilla Policy Gradient")
    vpg(lambda: gym.make(args.env), model_fn, model_kwargs, epochs=args.epochs,
        batch_size=args.batch_size, pi_lr=args.pi_lr, v_lr=args.v_lr, gamma=args.gamma,
        seed=args.seed, render=args.render, render_last=args.renderlast,
        logger_kwargs=logger_kwargs)
