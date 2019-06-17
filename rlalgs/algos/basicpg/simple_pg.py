"""
Simplest policy gradient algorithm

Components:
- Policy network
-

Based off of OpenAI's spinningup implementation of policy gradient

N.B. some of the code is extra verbose to help with understanding
"""
import gym
import numpy as np
import tensorflow as tf
import rlalgs.utils.utils as utils
import rlalgs.algos.basicpg.core as core
import rlalgs.utils.preprocess as preprocess
# from rlalgs.utils.logger import Logger

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def simplepg(env_fn, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000,
             seed=0, render=False, render_last=False):
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
    """

    print("Setting seeds")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("Initializing environment")
    env = env_fn()
    obs_dim = utils.get_dim_from_space(env.observation_space)
    num_actions = utils.get_dim_from_space(env.action_space)

    print("Initializing logger")
    # logger = Logger(output_fname="simple_pg" + env.spec._env_name + ".txt")

    print("Building network")
    # model = core.MLPCategoricalPolicy(num_actions, hidden_sizes)
    model = core.mlp(num_actions, hidden_sizes)

    print("Setup loss")

    def get_action(obs):
        logits = model.predict(obs)
        action = np.squeeze(tf.random.categorical(logits, 1), axis=1)
        return action

    def policy_loss(acts_and_rets, logits):
        """
        Uses cross entropy loss (Sparse since we input actions as an int rather than
        one hot encoding), where:

        y_true = action chosen
        y_pred = action probs
        weighted by trajectory return
        """
        actions, returns = tf.split(acts_and_rets, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)
        weighted_ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = weighted_ce_loss(actions, logits, sample_weight=returns)
        return loss

    print("Setting up training op")
    train_op = tf.keras.optimizers.Adam(learning_rate=lr)

    print("Compiling model")
    model.compile(optimizer=train_op, loss=tf.keras.losses.sparse_categorical_crossentropy)

    print("Initializing Replay Buffer")
    buf = core.SimpleBuffer(obs_dim, num_actions, batch_size, "simple")

    def train_one_epoch():
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
            # select action for current obs
            a = get_action(o.reshape(1, -1))
            # store step
            buf.store(o, a, r)
            # take action
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
                ep_len = 0
                batch_ep_rets.append(ep_ret)
                ep_ret = 0
                # finish epoch
                if t == batch_size:
                    break
        # get epoch trajectories
        batch_obs, batch_acts, batch_rets = buf.get()
        batch_loss = model.fit(batch_obs, batch_acts, sample_weight=batch_rets)
        return batch_loss, batch_ep_rets, batch_ep_lens

    print("Starting training")
    for i in range(epochs):
        batch_loss, batch_ep_rets, batch_ep_lens = train_one_epoch()
        print(f"\n{'-'*20}")
        print("epoch:", i)
        print("loss:", batch_loss.history)
        print("avg_return:", np.mean(batch_ep_rets))
        print("avg_ep_lens:", np.mean(batch_ep_lens))
        print(f"{'-'*20}\n")

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
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--renderlast", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("\nSimple Policy Gradient")
    simplepg(lambda: gym.make(args.env), epochs=args.epochs, lr=args.lr,
             seed=args.seed, render=args.render, render_last=args.renderlast)
