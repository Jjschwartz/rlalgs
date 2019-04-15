"""
This module contains MPI functions for A2C implementation
"""
import os
import sys
import subprocess
import numpy as np
from mpi4py import MPI
import tensorflow as tf


def mpi_fork(n, bind_to_core=False):
    """ Relaunch program using n processes """
    if n <= 1:
        return
    # check if function is being called from within an MPI forked process
    # since we are relaunching same script which calls this function
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        # set/update process environment variables
        # 1. set each process to be single-threaded
        # 2. set IN_MPI flag so this function is not recursively called by new process
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.run(args, env=env, check=True)
        sys.exit()


def proc_id():
    """ Get rank of calling process. """
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    """ Get number of processes running """
    return MPI.COMM_WORLD.Get_size()


def print_msg(msg, context=""):
    """ Print message including rank of process and optional context """
    print("Message from %d - %s: %s" % (proc_id(), context, str(msg)))


def sync_all_params(root=0):
    """ Sets all parameters across all processes to equal those of root process """
    global_vars = tf.global_variables()
    return sync_params(global_vars, root)


def sync_params(params, root=0):
    """ Syncs all the params across processes so they are equal to root's params """
    # 1 & 2. Flatten & Concat into one long 1D vector
    concat_vars = flat_concat(params)
    # 3. Broadcast
    synced_params = tf_broadcast(concat_vars, root)
    # 4. Get flattened sizes of OG variables
    flat_sizes = [int(np.prod(p.shape.as_list())) for p in params]
    # 5. Split Synced concatted 1D vector into OG flattened shapes
    splits = tf.split(synced_params, flat_sizes)
    # 6. Reshape synced flattened vectors into OG shapes
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    # 7. Assign synced variables to OG variables
    assigned = [tf.assign(p, p_new) for p, p_new in zip(params, new_params)]
    # 8. Group assign ops into one operation and return
    return tf.group(assigned)


def flat_concat(params):
    """ Flattens and concatenates a list of tensors  into a single 1D array """
    flattened_vars = [tf.reshape(x, (-1, )) for x in params]
    concat_vars = tf.concat(flattened_vars, axis=0)
    return concat_vars


def tf_broadcast(x, root=0):
    """ Creates function for syncing x from root process to all other processes """
    def _broadcast(x):
        broadcast(x)
        return x
    return tf.py_func(_broadcast, [x], tf.float32)


def broadcast(x, root=0):
    """ Sends x from root process to all other processes """
    MPI.COMM_WORLD.Bcast(x, root=root)


class MPIAdamOptimizer(tf.train.AdamOptimizer):
    """
    The AdamOptimizer which handles multiprocessor gradient descent:

    1. computes gradients for each process
    2. accumulates gradients from each process and takes average
    3. applies averaged gradients
    4. syncs all params across processes
    """

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Computes gradients, averaging gradients over processes
        """
        # 1. Get local gradients
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        # 2. Ignore variables that have no gradient since var_list, contains all train vars
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        # 3. Accumulate gradients across all processes
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        num_tasks = self.comm.Get_size()
        # buffer to store accumulated 1D grad tensor
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            # Sum grads across all processes
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            # Average by dividing by number of processes
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        # define the tf function for graph
        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)

        # 4. Reconstruct original grad tensors from accumulated 1D synced tensor
        # get sizes of OG grad tensors for reconstruction latter
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]
        # ensure correct flat shape
        avg_flat_grad.set_shape(flat_grad.shape)
        # split into OG flat tensor shapes
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        # reshape into OG shape and pair with OG var
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Apply gradients to model and synce model across all processes
        """
        # grads_and_vars are already accumulated from all processes, so just apply
        # to local model using normal AdamOptimizer
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        # control_dependencies just specifies that operation opt (i.e. apply gradients)
        # must be applied before the enclosed operations (i.e. syncing params)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g, v in grads_and_vars])
        return tf.group([opt, sync])
