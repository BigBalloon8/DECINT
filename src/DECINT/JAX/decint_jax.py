import jax
import optax
from mpi4py import MPI
import mpi4jax
import jax.numpy as jnp
import random
import DECINT_ai
import threads


# mpirun --mca btl

# TODO account for multiple gpus (use pmap)


@jax.jit
def grad_compress(grads, d_type="float16"):
    flat_grads = jax.tree_util.tree_flatten(grads)
    compressed_grads = [jax.tree_map(jax.vmap(lambda x: x.astype(d_type)), i) for i in flat_grads[0]]
    return compressed_grads


@jax.jit
def grad_uncompress(grads, compressed_grads, d_type="float32"):
    flat_grads = jax.tree_util.tree_flatten(grads)
    inter_flat_grads = [jax.tree_map(lambda x: x.astype(d_type), i) for i in compressed_grads]
    grads = jax.tree_unflatten(flat_grads[1], inter_flat_grads)
    return grads


@jax.jit
def ring_all_reduce(comm, grads, compression=True):
    # https://youtu.be/rj-hjS5L8Bw?t=1018 watch this not english but should still make sence
    """
    1. split grads into equal chunks dependent on comm.Get_size()
    2. send chunk[comm.Get_rank()] to (comm.Get_rank()+1 % comm.Get_size())
    3. receive chunk[(comm.Get_rank()-1 % comm.Get_size())]
    4. sum recieved chunk with local chunk[(comm.Get_rank()-1 % comm.Get_size())]
    5. repeat with chunk + 1 and so on till all comms have a sum all chunks then / by comm.Get_size()
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    if not compression:
        flat_grads = jax.tree_flatten(grads)
        grads = flat_grads[0]

    grads = jnp.array_split(grads, size)
    grads = jnp.concatenate((grads[:(rank + 1)], grads[(rank + 1):]))
    token = mpi4jax.barrier(comm=comm)

    for i, j in enumerate(grads):  # jaxify this

        if comm.Get_rank() == 0:
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)
            new_chunk, token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)
            grads[(i + 1) % size] = grads[(i + 1) % size] + new_chunk  # vmap this

        else:
            new_chunk, token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)
            grads[(i + 1) % size] = grads[(i + 1) % size] + new_chunk  # vmap this
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)

    grads = jnp.concatenate((grads[1:], grads[:1]))  # set completed arr to index 0

    for i, j in enumerate(grads):
        if comm.Get_rank() == 0:
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)
            grads[(i + 1) % size], token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)

        else:
            grads[(i + 1) % size], token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)

    grads = jnp.concatenate((grads[:(rank - 2) % size], grads[(rank - 2) % size:]))  # arrange back to norm
    grads = jax.vmap(lambda x: x / size)(grads)  # this wont work just temp

    if not compression:
        return jax.tree_unflatten(flat_grads[1], grads)
    return grads


def pair_average_opt(params, thread: threads.PairAverageProccess):
    """
    see Kung Fu docs https://kungfu.readthedocs.io/en/latest/?badge=latest
    Asynchronous Decentralized Parallel Stochastic Gradient Descent, ICML 2018
    https://arxiv.org/abs/1710.06952

    1. start Thread that waits for param request (run outside of function)
    2. after param update select random rank from comm (run outside of function)
    3. send a request and receive model params from selected rank
    4. average params with local params
    5. save params to thread to allow other processes to average with updated params
    """

    rank = thread.comm.Get_rank()
    size = thread.comm.Get_size()
    flat_params = jax.tree_flatten(params)
    thread["params"] = flat_params[0]
    while True:
        random_rank = random.randint(0, size - 1)  # TODO use jax random
        if random_rank != rank:
            break
    DECINT_ai.node.send(None, "GET_GRADS")  # TODO find away to pass rank IPs to pass ip
    if thread.compression:
        ranks_params, thread.token = mpi4jax.recv(grad_compress(flat_params[0], thread.comp_dtype), random_rank, comm=thread.comm, token=thread.token)
        ranks_params = grad_uncompress(ranks_params, thread.params_dtype)
    else:
        ranks_params, thread.token = mpi4jax.recv(flat_params[0], random_rank, comm=thread.comm, token=thread.token)
    return jax.tree_map(lambda x, y: (x + y) / 2., flat_params[0], ranks_params)


def decint_opt(thread: threads.DecintMpiThread, grads, params, opt, opt_state):
    thread.put_grads(grads)
    grads = thread["grads"]
    updates, opt_state = opt.update(grads, opt_state, params)
    params = opt.apply_updates(updates, params)
    return params, opt_state

d
