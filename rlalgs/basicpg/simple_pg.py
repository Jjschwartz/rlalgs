"""
Simplest policy gradient algorithm

Components:
- Policy network
-

Based off of OpenAI's spinningup implementation of policy gradient

N.B. some of the code is extra verbose to help with understanding
"""
import numpy as np
import gym
import tensorflow as tf



class ReplayBuffer:
    """
    A buffer for storing trajectories (o, a, r)
    """

    def __init__(self, obs_dim, act_dim, buffer_size):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, o, a, r):
        """
        Store a step outcome (o, a, r) in the buffer
        """
        self.obs_buf.append(o)
        self.act_buf.append(a)
        self.rew_buf.append(r)
        self.ptr += 1

    def finish_path(self):
        """
        Called when an episode is done
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        ep_len = self.ptr - self.path_start_idx
        ep_rews = self.rew_buf[path_slice]
        ep_ret = np.sum(ep_rews)
        self.ret_buf += [ep_ret] * ep_len
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return the stored trajectories and empty the buffer
        """
        self.ptr, self.path_start_idx = 0, 0
        obs = self.obs_buf
        acts = self.act_buf
        rets = self.ret_buf
        self.obs_buf, self.act_buf, self.rew_buf, self.ret_buf = [], [], [], []
        return obs, acts, rets

    def size(self):
        """
        Return size of buffer
        """
        return self.ptr


def simplepg(env_fn, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000,
             seed=0, render=False):
    """
    Simple Policy Gradient

    Only works for continuous observation space and discrete action space

    Arguments:
    ----------
        env_fn : A function which creates a copy of OpenAI Gym environment

        hidden_sizes : list of units in each hidden layer of policy network

        lr : learning rate for policy network update

        epochs : number of epochs to train for

        batch_size : max batch size for epoch

        seed : random seed

        render : whether to render environment or not
    """
    print("Setting seeds")
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # instantiate environment
    print("Initializing env")
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = get_action_dim(env.action_space)

    # build policy network
    print("Building network")
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, hidden_sizes=hidden_sizes+[act_dim])

    # random action selection based off raw probabilities
    actions = tf.squeeze(tf.multinomial(logits, 1), axis=1)

    # build loss function
    print("Building loss function")
    return_ph = tf.placeholder(tf.float32, shape=(None, ))  # takes batch of trajectory return
    act_ph = tf.placeholder(tf.int32, shape=(None, ))   # takes batch of trajectory actions
    action_mask = tf.one_hot(act_ph, act_dim)   # converts chosen action into one hot vector
    # Calculate the log probability for each action taken in trajectory
    # log probability = log_prob of action if acton taken otherwise 0 (hence action mask)
    log_probs = action_mask * tf.nn.log_softmax(logits)
    # sum log probs for a given trajectory
    log_probs_sum = tf.reduce_sum(log_probs, axis=1)
    # take the mean over all trajectories of trajectory probability weighted by return
    policy_grad = tf.reduce_mean(log_probs_sum * return_ph)
    # loss is negative of policy gradient
    loss = -policy_grad

    # training operation
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Trajectory buffer
    buf = ReplayBuffer(obs_dim, act_dim, batch_size)

    # initialize tf session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def train_one_epoch():

        o, r, d = env.reset(), 0, False

        finished_rendering_this_epoch = False

        while True:

            # render first episode of each epoch
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # select action for current obs
            a = sess.run(actions, {obs_ph: o.reshape(1, -1)})[0]

            # store step
            buf.store(o, a, r)

            # take action
            o, r, d, _ = env.step(a)

            # end of episode
            if d:
                buf.finish_path()

                o, r, d = env.reset(), 0, False

                finished_rendering_this_epoch = True

                # finish epoch
                if buf.size() > batch_size:
                    break

        # get epoch trajectories
        batch_obs, batch_acts, batch_rets = buf.get()

        # take single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    return_ph: np.array(batch_rets)
                                 })

        return batch_loss, batch_rets

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets = train_one_epoch()
        print("epoch: %3d \t loss: %.3f \t return: %.3f" %
              (i, batch_loss, np.mean(batch_rets)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    print("\nSimple Policy Gradient")

    simplepg(lambda: gym.make(args.env), lr=args.lr, render=args.render)
