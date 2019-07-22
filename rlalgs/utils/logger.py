"""
Logger class

Logs training progress to a tab-seperated values file
Also provides functionality for saving and restoring a model

Inspired heavily by OpenAI spinningup logger, which was in turn inspired by rllab's logging.
"""
import os
import gym
import json
import atexit
import shutil
import pickle
import os.path as osp

from tensorflow.keras.models import load_model

import rlalgs.algos.policy as policy_fn
from rlalgs.utils.serialization_utils import convert_json


DEFAULT_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
OBS_NAME = "x"
ACTS_NAME = "pi"

# some constants to make saving and loading models easier
ACTOR_MODEL_NAME = "policy"
CRITIC_MODEL_NAME = "value"


def setup_logger_kwargs(exp_name, data_dir=None, seed=None, verbose=True):
    """
    Set up output directory for logger.

    if seed is None:
        output_dir = data_dir/exp_name
    else:
        output_dir = data_dir/exp_name/exp_name_s[seed]

    Arguments:
        exp_name : name for experiment
        data_dir : path to folder where results should be saved
            if None uses DEFAULT_DIR
        seed : seed for random number generators used by experiment
        verbose : whether logger should output performance metrics each epoch

    Returns:
        logger_kwargs : dictionary containing output_dir and exp_name
    """
    if data_dir is None:
        data_dir = DEFAULT_DIR

    if seed is not None:
        subfolder = ''.join([exp_name, '_s', str(seed)])
        expfolder = osp.join(exp_name, subfolder)
    else:
        expfolder = exp_name
    output_dir = osp.join(data_dir, expfolder)

    return dict(output_dir=output_dir, exp_name=exp_name, verbose=verbose)


def restore_model(model_dir):
    """
    Loads a model from directory.

    Arguments:
        str model_dir : model directory path

    Returns:
        pi_model : tf.Keras policy/actor model
        v_model : tf.Keras value/critic model (may be None, depending on algorithm)
        pi_fn : the action selection function
    """

    with open(osp.join(model_dir, "exp_info.pkl"), "rb") as info_file:
        info = pickle.load(info_file)

    models = {}
    for model_name in info["model_names"]:
        models[model_name] = load_model(osp.join(model_dir, model_name + ".h5"))

    pi_model = models[ACTOR_MODEL_NAME]
    v_model = models.get(CRITIC_MODEL_NAME)

    env = gym.make(info["env"])
    alg_type = info["alg_type"]
    pi_fn_wrapper = policy_fn.get_policy_fn(env, alg_type)
    pi_fn = pi_fn_wrapper(pi_model)

    return pi_model, v_model, pi_fn


def get_env_name(model_dir):
    with open(osp.join(model_dir, "exp_info.pkl"), "rb") as info_file:
        info = pickle.load(info_file)
    return info["env"]


class Logger:
    """
    A simple logger

    Allows for:
    - storing and writing tabular statistics to a tab seperated file

    Simply call log_tabular(diagnostic_name, diagnostic_value) to store a key-value pair.
    Then call dump_tabular() to write all stored key-value pairs to tab seperated file.

    """

    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None, verbose=True):
        """
        Initialize logger to write to output_dir/output_file

        Arguments:
            str output_dir : the directory to save file. If None then DEFAULT_DIR is
                             used
            str output_fname : the name of output file
            str exp_name : name of experiment
            bool verbose : whether to print detailed info or not
        """
        self.output_dir = DEFAULT_DIR if output_dir is None else output_dir
        if osp.exists(self.output_dir):
            print("Warning: Log dir {} already exists, but storing there anyway ;)".format(self.output_dir))
        else:
            os.makedirs(self.output_dir)
        self.output_fname = osp.join(self.output_dir, output_fname)
        self.output_file = open(self.output_fname, "w", buffering=1)
        # closes file when module exits
        atexit.register(self.output_file.close)
        self.first_row = True
        self.headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.verbose = verbose

    def save_config(self, config):
        """
        Saves the configuration (env, hyperparams, etc) for a given algorithm run.
        Configuration is saved into a file called "config.json" in output directory.

        Arguments:
            dict config : local config dictionary of algorithm
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name
        if 'logger' in config_json:
            del config_json['logger']
        if self.verbose:
            print("\nSaving config:\n")
            for k, v in config_json.items():
                print("\t{}: {}".format(k, v))
            print("\n")
        with open(osp.join(self.output_dir, "config.json"), "w") as out:
            json.dump(config_json, out, separators=(',', ':\t'), indent=2, sort_keys=True)

    def save_model(self, itr=None):
        """
        Save the current model.

        If itr is not None saves model to new directory, otherwise rewrites old saved model if one
        exists.
        """
        assert hasattr(self, "tf_saver_elements"), \
            "First have to setup model saving with self.setup_tf_model_saver, before saving model"
        base_model_dir = self.tf_saver_elements["base_model_dir"]
        model_dir = base_model_dir if itr is None else base_model_dir + str(itr)
        if osp.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        # save models
        for model_name, model in self.tf_saver_elements["models"].items():
            model.save(osp.join(model_dir, model_name + ".h5"))
        # save model info
        with open(osp.join(model_dir, "exp_info.pkl"), "wb") as info_file:
            pickle.dump(self.tf_model_info, info_file)

    def setup_tf_model_saver(self, pi_model, env, alg_type, v_model=None):
        """
        Set up model saver info.
        This should be called before save_model.

        Arguments:
            pi_model : the policy/actor model
            env : gym environment
            alg_type : the algorithm type ("pg", "ql")
            v_model : the value/critic model (default is None, if not saving critic or
                algorithm doesn't use one)
        """
        assert alg_type in policy_fn.VALID_ALG_TYPES
        base_model_dir = osp.join(self.output_dir, "simple_save")
        models = {ACTOR_MODEL_NAME: pi_model}
        if v_model is not None:
            models[CRITIC_MODEL_NAME] = v_model
        self.tf_saver_elements = dict(models=models, base_model_dir=base_model_dir)
        self.tf_model_info = {'env': env.spec.id,
                              'alg_type': alg_type,
                              'model_names': list(models.keys())}

    def log_tabular(self, key, value):
        """
        Log value of some diagnostic

        If it is the first row being logged, then the key is added as a header and
        subsequent logs with that key are added under this header.
        If it is not the first row and the key is not as header, then n error is
        raised.

        Arguments:
            str key : diagnostic name
            misc val : value of diagnostic
        """
        if self.first_row:
            self.headers.append(key)
        else:
            assert key in self.headers, "Trying to log a new key %s after the first row" % key
        assert key not in self.log_current_row, "Value for key %s already logged for this iteration" % key  # noqa E501
        self.log_current_row[key] = value

    def dump_tabular(self):
        """
        Write all stored diagnostic-value pairs to output file and stdout, and clear buffer.
        """
        vals = []
        max_header_len = max(12, len(max(self.headers)))

        if self.verbose:
            print("\n{}".format("-"*2*(max_header_len + 8)))

        for k in self.headers:
            v = self.log_current_row[k]
            num_spaces = max_header_len - len(k)
            vstr = "%.3g" % v if isinstance(v, float) else str(v)
            if self.verbose:
                print("| {}{} \t\t {}".format(" " * num_spaces, k, vstr))
            vals.append(vstr)

        if self.verbose:
            print("{}".format("-"*2*(max_header_len + 8)))

        if self.first_row:
            self.output_file.write("\t".join(self.headers) + "\n")

        self.output_file.write("\t".join(vals) + "\n")
        self.log_current_row.clear()
        self.first_row = False

    def get_stats(self, key):
        """
        Get a specific statistic from the logger.

        Only returns the latest row, so calling this directly after dump_tabular will return
        nothing.
        """
        return self.log_current_row[key]

    def get_output_filename(self):
        return self.output_fname
