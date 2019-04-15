"""
Logger class

Logs training progress to a tab-seperated values file
Also provides functionality for saving and restoring a model

Inspired heavily by OpenAI spinningup logger, which was in turn inspired by rllab's logging.
"""
import os
import atexit
import shutil
import pickle
import os.path as osp
import tensorflow as tf


DEFAULT_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
OBS_NAME = "x"
ACTS_NAME = "pi"


def setup_logger_kwargs(exp_name, data_dir=None, seed=None):
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

    return dict(output_dir=output_dir, exp_name=exp_name)


def save_model(sess, output_dir, env, inputs, outputs):
    model_dir = osp.join(output_dir, "simple_save")
    if osp.exists(model_dir):
        shutil.rmtree(model_dir)
    saver = tf.train.Saver()
    saver.save(sess, osp.join(model_dir, "model"))

    info_file = open(osp.join(model_dir, "exp_info.pkl"), "wb")
    info = {"env": env.spec.id,
            "inputs": {k: v.name for k, v in inputs.items()},
            "outputs": {k: v.name for k, v in outputs.items()}}
    pickle.dump(info, info_file)
    info_file.close()


def restore_model(sess, model_dir):
    saver = tf.train.import_meta_graph(osp.join(model_dir, "model" + ".meta"))
    saver.restore(sess, osp.join(model_dir, "model"))
    graph = tf.get_default_graph()

    info_file = open(osp.join(model_dir, "exp_info.pkl"), "rb")
    info = pickle.load(info_file)
    info_file.close()

    model_vars = dict()
    model_vars["inputs"] = {k: graph.get_tensor_by_name(v) for k, v in info["inputs"].items()}
    model_vars["outputs"] = {k: graph.get_tensor_by_name(v) for k, v in info["outputs"].items()}
    return model_vars


def get_env_name(model_dir):
    info_file = open(osp.join(model_dir, "exp_info.pkl"), "rb")
    info = pickle.load(info_file)
    info_file.close()
    return info["env"]


class Logger:
    """
    A simple logger

    Allows for:
    - storing and writing tabular statistics to a tab seperated file

    Simply call log_tabular(diagnostic_name, diagnostic_value) to store a key-value pair.
    Then call dump_tabular() to write all stored key-value pairs to tab seperated file.

    """

    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
        """
        Initialize logger to write to output_dir/output_file

        Arguments:
            str output_dir : the directory to save file. If None then DEFAULT_DIR is
                             used
            str output_fname : the name of output file
        """
        self.output_dir = DEFAULT_DIR if output_dir is None else output_dir
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists, but storing there anyway ;)")
        else:
            os.makedirs(self.output_dir)
        self.output_fname = osp.join(self.output_dir, output_fname)
        self.output_file = open(self.output_fname, "w")
        # closes file when module exits
        atexit.register(self.output_file.close)
        self.first_row = True
        self.headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

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
        max_header_len = max(10, len(max(self.headers)))
        print("\n{}".format("-"*2*(max_header_len + 8)))
        for k in self.headers:
            v = self.log_current_row[k]
            num_spaces = max_header_len - len(k)
            vstr = "%.3g" % v if isinstance(v, float) else str(v)
            print("| {}{} \t\t {}".format(" " * num_spaces, k, vstr))
            vals.append(vstr)
        if self.first_row:
            self.output_file.write("\t".join(self.headers) + "\n")
        self.output_file.write("\t".join(vals) + "\n")
        print("{}".format("-"*2*(max_header_len + 8)))
        self.log_current_row.clear()
        self.first_row = False

    def get_output_filename(self):
        return self.output_fname
