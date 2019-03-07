"""
Logger class

Logs training progress to a tab-seperated values file
Also provides functionality for saving and restoring a model

Inspired heavily by OpenAI spinningup logger, which was in turn inspired by rllab's logging.
"""
import os.path as osp
import atexit
import shutil
import tensorflow as tf
import pickle


DEFAULT_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
OBS_NAME = "x"
ACTS_NAME = "pi"


def save_model(sess, save_name, env, inputs, outputs):
    output_dir = osp.join(DEFAULT_DIR, save_name + "/")
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    saver = tf.train.Saver()
    saver.save(sess, output_dir + save_name)

    info_file = open(osp.join(output_dir + "exp_info.pkl"), "wb")
    info = {"env": env.spec.id,
            "inputs": {k: v.name for k, v in inputs.items()},
            "outputs": {k: v.name for k, v in outputs.items()}}
    pickle.dump(info, info_file)
    info_file.close()


def restore_model(sess, save_name):
    output_dir = osp.join(DEFAULT_DIR, save_name + "/")
    saver = tf.train.import_meta_graph(output_dir + save_name + ".meta")
    saver.restore(sess, output_dir + save_name)
    graph = tf.get_default_graph()

    info_file = open(osp.join(output_dir + "exp_info.pkl"), "rb")
    info = pickle.load(info_file)
    info_file.close()

    model_vars = dict()
    model_vars["inputs"] = {k: graph.get_tensor_by_name(v) for k, v in info["inputs"].items()}
    model_vars["outputs"] = {k: graph.get_tensor_by_name(v) for k, v in info["outputs"].items()}
    return model_vars


def get_env_name(save_name):
    output_dir = osp.join(DEFAULT_DIR, save_name + "/")
    info_file = open(osp.join(output_dir + "exp_info.pkl"), "rb")
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

    def __init__(self, output_dir=None, output_fname="progress.txt"):
        """
        Initialize logger to write to output_dir/output_file

        Arguments:
            str output_dir : the directory to save file. If None then DEFAULT_DIR is
                             used
            str output_fname : the name of output file
        """
        self.output_dir = DEFAULT_DIR if output_dir is None else output_dir
        self.output_file = open(osp.join(self.output_dir, output_fname), "w")
        # closes file when module exits
        atexit.register(self.output_file.close)
        self.first_row = True
        self.headers = []
        self.log_current_row = {}

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
