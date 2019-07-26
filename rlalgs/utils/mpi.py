"""
This module contains MPI functions for A2C implementation
"""
import os
import sys
import subprocess
import numpy as np
from mpi4py import MPI


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


def broadcast(x, root=0):
    """Sends x from root process to all other processes """
    MPI.COMM_WORLD.Bcast(x, root=root)
    return x


def print_msg(msg, context=""):
    """ Print message including rank of process and optional context """
    print("Message from %d - %s: %s" % (proc_id(), context, str(msg)))


def flat_concat(params):
    """ Flattens and concatenates a list of tensors  into a single 1D array """
    flattened_vars = [np.reshape(x, (-1, )) for x in params]
    concat_vars = np.concatenate(flattened_vars, axis=0)
    return concat_vars


def sync_all_params(params, root=0):
    """ Sets all parameters across all processes to equal those of root process """
    return sync_params(params, root)


def sync_params(params, root=0):
    """ Syncs all the params across processes so they are equal to root's params """
    # 1 & 2. Flatten & Concat into one long 1D vector
    concat_vars = flat_concat(params)
    # 3. Broadcast
    synced_params = broadcast(concat_vars, root)
    # 4. Get flattened sizes of OG variables
    flat_sizes = [int(np.prod(p.shape)) for p in params]
    flat_sizes = np.cumsum(flat_sizes)
    # 5. Split Synced concatted 1D vector into OG flattened shapes
    splits = np.split(synced_params, flat_sizes, axis=0)
    # 6. Reshape synced flattened vectors into OG shapes
    new_params = [np.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return new_params


def sync_gradients(grads):
    """Sync and average gradients across processes """
    num_tasks = num_procs()
    # Convert grads into a single flat tensor
    flat_grads = flat_concat([g for g in grads])
    # buffer to store accumulated 1D grad tensor
    buf = np.zeros(flat_grads.shape, np.float32)
    # Sum grads across all processes
    MPI.COMM_WORLD.Allreduce(flat_grads, buf, op=MPI.SUM)
    # Average by dividing by number of processes
    avg_flat_grads = np.divide(buf, float(num_tasks))
    # Reconstruct original grad tensors from accumulated 1D synced tensor
    # get sizes of OG grad tensors for reconstruction latter
    sizes = [int(np.prod(g.shape)) for g in grads]
    sizes = np.cumsum(sizes)
    # ensure correct flat shape
    avg_flat_grads.reshape(flat_grads.shape)
    # split into OG flat tensor shapes
    avg_grads_1D = np.split(avg_flat_grads, sizes, axis=0)
    # reshape into OG 2D tensor shapes
    avg_grads_correct = [np.reshape(a, g.shape) for a, g in zip(avg_grads_1D, grads)]
    return avg_grads_correct
