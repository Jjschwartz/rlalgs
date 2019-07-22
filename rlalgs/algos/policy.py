import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

POLICY_GRAD = "pg"
Q_LEARNING = "ql"
VALID_ALG_TYPES = [POLICY_GRAD, Q_LEARNING]


def get_policy_fn(env, alg_type):
    """Get the policy function given the environment and the algorithm type. """
    if alg_type == "ql":
        return discrete_pg

    action_space = env.action_space
    if isinstance(action_space, Box):
        return continuous_pg
    elif isinstance(action_space, Discrete):
        return discrete_pg
    else:
        raise NotImplementedError


def discrete_pg(model):
    """Policy method for discrete actions using policy gradient. """
    @tf.function
    def model_query(o):
        logits = model(o)
        return tf.squeeze(tf.random.categorical(logits, 1), axis=1)

    def act_fn(o):
        a_tensor = model_query(o.reshape(1, -1))
        return np.squeeze(a_tensor, axis=-1)

    return act_fn


def discrete_qlearning(model):
    """Policy method for discrete actions using Q-learning. """
    @tf.function
    def model_query(o):
        a = tf.argmax(model(o), axis=1)
        return tf.squeeze(a)

    def act_fn(o):
        a_tensor = model_query(o.reshape(1, -1))
        return np.squeeze(a_tensor, axis=-1)

    return act_fn


def continuous_pg(model, log_std=-0.5):
    """Policy method for continuous actions using policy gradient. """
    log_std = log_std*np.ones(tf.shape(model.output))
    std = tf.exp(log_std)

    @tf.function
    def model_query(o):
        return model(o) + tf.random.normal(tf.shape(model.output)) * std

    def act_fn(o):
        a_tensor = model_query(o.reshape(1, -1))
        return a_tensor.numpy()

    return act_fn
