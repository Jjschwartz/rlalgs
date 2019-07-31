"""
Deep Q-Network implementation using tensorflow
"""
import gym
import time
import numpy as np
from gym.spaces import Discrete

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers

import rlalgs.utils.logger as log
import rlalgs.utils.utils as utils
import rlalgs.algos.dqn.core as core
import rlalgs.utils.preprocess as preprocess
from rlalgs.algos.models import mlp_q_network, cnn_q_network


def dqn(env_fn, model_fn, model_kwargs, lr=1e-3, epochs=50, epoch_steps=10000, batch_size=32,
        seed=0, replay_size=100000, epsilon=0.05, gamma=0.99, polyak=0.995, start_steps=100000,
        target_update_freq=1, render=False, render_last=False, logger_kwargs=dict(), save_freq=10,
        overwrite_save=True, preprocess_fn=None, obs_dim=None):
    """Deep Q-network with experience replay

    Arguments:
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    model_fn : function for creating the Q-network model to use (see models module for more info)
    model_kwargs : any kwargs to pass into model function
    lr : learning rate for policy network update
    epochs : number of epochs to train for
    epoch_steps : number of steps per epoch
    batch_size : number of steps between main network updates
    seed : random seed
    replay_size : max size of replay buffer
    epsilon : random action selection parameter
    gamma : discount parameter
    polyak : Interpolation factor when copying target network towards main network.
        (set to 0.0 if wanting to use n-step target network updating)
    start_steps : the epsilon annealing period in number of steps
    target_update_freq : number of steps between target network updates
        (should be one if using polyak averaging or <= epoch_steps for n-step updating)
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
    assert target_update_freq <= epoch_steps, \
        "must have target_update_freq <= epoch_steps, else no learning will be done.."

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
    if not isinstance(env.action_space, Discrete):
        raise NotImplementedError("DQN only works for environments with Discrete action spaces")

    if obs_dim is None:
        obs_dim = env.observation_space.shape
    num_actions = utils.get_dim_from_space(env.action_space)
    act_dim = env.action_space.shape

    print("Initializing buffer")
    buf = core.DQNReplayBuffer(obs_dim, act_dim, replay_size)

    epsilon_schedule = np.linspace(1, epsilon, start_steps)
    global total_t
    total_t = 0

    print("Building network")
    # main network
    q_model, pi_fn, q_fn = model_fn(env, **model_kwargs)
    # target network
    q_model_targ, pi_fn_targ, q_fn_targ = model_fn(env, **model_kwargs)

    print("Setting up training ops")
    q_optimizer = optimizers.Adam(learning_rate=lr)

    @tf.function
    def q_loss(target_q_val, act_q_val, done, rew):
        target = rew + gamma*(1-done)*target_q_val
        return tf.reduce_mean((tf.stop_gradient(target) - act_q_val)**2)

    @tf.function
    def get_grads_and_apply(obs, obs_prime, acts, rews, dones):
        with tf.GradientTape() as tape:
            target_q_val, _ = q_fn_targ(acts, obs_prime)
            _, act_q_val = q_fn(acts, obs)
            loss = q_loss(target_q_val, act_q_val, dones, rews)
        grads = tape.gradient(loss, q_model.trainable_variables)
        q_optimizer.apply_gradients(zip(grads, q_model.trainable_variables))
        return loss

    print("Setting up main and target network copying")

    def copy_network_weights():
        # update target network to match main network
        q_model_targ.set_weights(q_model.get_weights())

    copy_network_weights()

    def update(t):
        batch = buf.sample(batch_size)
        loss = get_grads_and_apply(batch['o'], batch["o_prime"], batch["a"], batch["r"], batch["d"])

        if t > 0 and (target_update_freq == 1 or t % (target_update_freq-1) == 0):
            # if t == epoch_steps-1:
            #     logger.log_tabular("ntwk_diff", network_diff())
            copy_network_weights()
        return loss.numpy()

    print("Setting up model saver")
    logger.setup_tf_model_saver(q_model, env, "ql")

    def get_action(o, t):
        eps = epsilon if t >= start_steps else epsilon_schedule[t]
        if np.random.rand(1) < eps:
            a = np.random.choice(num_actions)
        else:
            a = pi_fn(o)
        return np.squeeze(a, axis=-1)

    def network_diff():
        """ Calculates difference between networks """
        main_var = q_model.get_weights()
        target_var = q_model_targ.get_weights()
        total_diff = 0
        for m_val, t_val in zip(main_var, target_var):
            diff = np.sum(np.abs(m_val - t_val))
            total_diff += diff
        return total_diff

    def train_one_epoch():
        global total_t
        o, r, d = env.reset(), 0, False
        finished_rendering_this_epoch = False
        ep_len, ep_ret, ep_loss = 0, 0, []
        epoch_ep_lens, epoch_ep_rets, epoch_ep_loss = [], [], []
        t = 0

        o = preprocess_fn(o, env)
        while True:
            if not finished_rendering_this_epoch and render:
                env.render()

            a = get_action(o, total_t)
            o_prime, r, d, _ = env.step(a)
            o_prime = preprocess_fn(o_prime, env)
            buf.store(o, a, r, o_prime, d)

            batch_loss = update(t)
            ep_len += 1
            ep_ret += r
            t += 1
            total_t += 1
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
        logger.log_tabular("epoch", i)
        epoch_start = time.time()
        results = train_one_epoch()
        epoch_time = time.time() - epoch_start

        total_epoch_times += epoch_time
        total_episodes += len(results[2])
        training_time_left = utils.training_time_left(i, epochs, epoch_time)

        logger.log_tabular("q_loss", np.mean(results[0]))
        logger.log_tabular("avg_return", np.mean(results[1]))
        logger.log_tabular("avg_ep_lens", np.mean(results[2]))
        logger.log_tabular("total_eps", total_episodes)
        logger.log_tabular("end_epsilon", epsilon if total_t >= start_steps else epsilon_schedule[total_t])
        logger.log_tabular("epoch_time", epoch_time)
        logger.log_tabular("mem_usage", utils.get_current_mem_usage())
        logger.log_tabular("time_rem", training_time_left)

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
    parser.add_argument("--epoch_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--start_steps", type=int, default=100000)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    exp_name = "dqn_" + args.env if args.exp_name is None else args.exp_name
    logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)

    preprocess_fn, obs_dim = preprocess.get_preprocess_fn(args.env)

    model_fn = mlp_q_network
    if args.model == "cnn":
        model_fn = cnn_q_network
    model_kwargs = {"hidden_sizes": args.hidden_sizes}

    print("\nDeep Q-Network")
    dqn(lambda: gym.make(args.env), model_fn, model_kwargs, lr=args.lr, epochs=args.epochs,
        epoch_steps=args.epoch_steps, batch_size=args.batch_size, seed=args.seed,
        replay_size=args.replay_size, epsilon=args.epsilon, gamma=args.gamma, polyak=args.polyak,
        start_steps=args.start_steps, target_update_freq=args.target_update_freq,
        render=args.render, render_last=args.renderlast, logger_kwargs=logger_kwargs,
        preprocess_fn=preprocess_fn, obs_dim=obs_dim)
