"""
Synchronous Advantage Actor-Critic (A2C) implementation
"""
import gym
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers

import rlalgs.utils.mpi as mpi
import rlalgs.utils.logger as log
import rlalgs.utils.utils as utils
import rlalgs.utils.preprocess as preprocess
from rlalgs.algos.buffers import PGReplayBuffer
from rlalgs.algos.models import mlp_actor_critic, cnn_actor_critic, print_model_summary


def a2c(env_fn, model_fn, model_kwargs, epochs=50, steps_per_epoch=5000, pi_lr=1e-2, vf_lr=1e-2,
        gamma=0.99, seed=0, logger_kwargs=dict(), save_freq=10,
        overwrite_save=True, preprocess_fn=None, obs_dim=None):
    """
    Train agent on env using A2C

    Arguments
    ----------
    env_fn : A function which creates a copy of OpenAI Gym environment
    model_fn : function for creating the policy gradient models to use
        (see models module for more info)
    model_kwargs : any kwargs to pass into model function
    hidden_sizes : list of units in each hidden layer of policy network
    epochs : number of epochs to train for
    steps_per_epoch : number of steps per epoch
    pi_lr : learning rate for policy network update
    vf_lr : learning rate for value network update
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
        obs_dim = env.observation_space.shape
    num_actions = utils.get_dim_from_space(env.action_space)
    act_dim = env.action_space.shape

    mpi.print_msg("Initializing Replay Buffer")
    local_steps_per_epoch = int(steps_per_epoch / mpi.num_procs())
    buf = PGReplayBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma=gamma)

    mpi.print_msg("Building network")
    pi_model, pi_fn, v_model, v_fn = model_fn(env, **model_kwargs)

    if mpi.proc_id() == 0:
        print_model_summary({"Actor": pi_model, "Critic": v_model})

    mpi.print_msg("Setup training ops - actor")
    pi_train_op = optimizers.Adam(learning_rate=pi_lr)

    @tf.function
    def policy_loss(a_pred, a_taken, a_adv):
        action_mask = tf.one_hot(tf.cast(a_taken, tf.int32), num_actions)
        log_probs = tf.reduce_sum(action_mask * tf.nn.log_softmax(a_pred), axis=1)
        return -tf.reduce_mean(log_probs * a_adv)

    mpi.print_msg("Setup training ops - critic")
    v_train_op = optimizers.Adam(learning_rate=vf_lr)

    @tf.function
    def value_loss(o_val, o_ret):
        return tf.reduce_mean((o_ret - o_val)**2)

    # 9. Sync all params across processes
    mpi.print_msg("Syncing all params")

    def sync():
        new_pi_weights = mpi.sync_params(pi_model.get_weights())
        pi_model.set_weights(new_pi_weights)
        new_v_weights = mpi.sync_params(v_model.get_weights())
        v_model.set_weights(new_v_weights)

    sync()

    if mpi.proc_id() == 0:
        # only save model of one cpu
        logger.setup_tf_model_saver(pi_model, env, "pg", v_model)

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

    def update():
        batch_obs, batch_acts, batch_rets, batch_adv, batch_vals = buf.get()
        pi_loss, pi_grads, v_loss, v_grads = get_grads(
            batch_obs, batch_acts, batch_rets, batch_adv)
        avg_pi_grads = mpi.sync_gradients(pi_grads)
        avg_v_grads = mpi.sync_gradients(v_grads)
        apply_gradients(avg_pi_grads, avg_v_grads)
        apply_gradients(pi_grads, v_grads)
        sync()
        return pi_loss, v_loss

    # 9. The training loop
    def train_one_epoch():
        o, r, d = env.reset(), 0, False
        batch_ep_rets, batch_ep_lens = [], []
        ep_ret, ep_len = 0, 0

        for t in range(local_steps_per_epoch):
            o = preprocess_fn(o, env)
            a = pi_fn(o)
            v_t = v_fn(o)
            buf.store(o, a, r, v_t)
            o, r, d, _ = env.step(a)

            ep_len += 1
            ep_ret += r

            if d or t == local_steps_per_epoch-1:
                if d:
                    last_val = r
                    batch_ep_rets.append(ep_ret)
                    batch_ep_lens.append(ep_len)
                else:
                    o = preprocess_fn(o, env)
                    last_val = v_fn(o)
                buf.finish_path(last_val)

                o, r, d = env.reset(), 0, False
                ep_ret, ep_len = 0, 0

        pi_loss, v_loss = update()
        return pi_loss.numpy(), v_loss.numpy(), batch_ep_rets, batch_ep_lens

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

            if (save_freq != 0 and epoch % save_freq == 0) or epoch == epochs-1:
                itr = None if overwrite_save else epoch
                logger.save_model(itr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("-m", "--model", type=str, default="mlp")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--pi_lr", type=float, default=1e-3)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    mpi.mpi_fork(args.cpu)
    mpi.print_msg("Running A2C!")

    exp_name = f"a2c_{args.cpu}_{args.env}" if args.exp_name is None else args.exp_name
    if mpi.proc_id() == 0:
        logger_kwargs = log.setup_logger_kwargs(exp_name, seed=args.seed)
    else:
        logger_kwargs = dict()

    model_fn = mlp_actor_critic
    if args.model == "cnn":
        model_fn = cnn_actor_critic
    model_kwargs = {"hidden_sizes": args.hidden_sizes, "share_layers": True}

    a2c(lambda: gym.make(args.env), model_fn, model_kwargs, epochs=args.epochs,
        steps_per_epoch=args.steps, pi_lr=args.pi_lr, vf_lr=args.vf_lr, seed=args.seed,
        gamma=args.gamma, logger_kwargs=logger_kwargs)
