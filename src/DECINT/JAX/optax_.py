import jax
from mpi4py import MPI
import mpi4jax
import jax.numpy as jnp

#  mpirun --mca btl 

#TODO account for multiple gpus (use pmap)

@jax.jit
def grad_compress(grads, d_type="float16"):
    flat_grads = jax.tree_util.tree_flatten(grads)
    grads_d_type = str(flat_grads[0][0].dtype)
    compressed_grads = [jax.tree_map(jax.vmap(lambda x : x.astype(d_type)), i) for i in flat_grads[0]]
    return compressed_grads, grads_d_type


@jax.jit
def grad_uncompress(grads, compressed_grads, d_type="float32"):
    flat_grads = jax.tree_util.tree_flatten(grads)
    inter_flat_grads = [jax.tree_map(lambda x: x.astype(d_type), i) for i in compressed_grads]
    grads = jax.tree_unflatten(flat_grads[1], inter_flat_grads)
    return grads

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
    size = comm.Get_size
    token = 0
    if not compression:
        flat_grads = jax.tree_flatten(grads)
        grads = flat_grads[0]

    grads = jnp.array_split(grads, size)
    grads = jnp.concatenate((grads[:(rank+1)], grads[(rank+1):]))

    for i, j in enumerate(grads):  # jaxify this

        if comm.Get_rank() == 0:
            if token:
                token = mpi4jax.send(j, dest=((rank+1) % size), comm=comm, token=token)
            else:
                token = mpi4jax.send(j, dest=((rank+1) % size), comm=comm)
            new_chunk, token = mpi4jax.recv(j, source=((rank-1) % size), comm=comm, token=token)
            grads[(i+1) % size] = grads[(i+1) % size] + new_chunk  # vmap this

        else:
            if token:  # TODO have to check if this works
                new_chunk, token = mpi4jax.recv(j, source=((rank-1) % size), comm=comm, token=token)
            else:
                new_chunk, token = mpi4jax.recv(j, source=((rank-1) % size), comm=comm)
            grads[(i+1) % size] = grads[(i+1) % size] + new_chunk  # vmap this
            token = mpi4jax.send(j, dest=((rank+1) % size), comm=comm, token=token)

    grads = jnp.concatenate((grads[1:], grads[:1]))  # set completed arr to index 0

    for i, j in enumerate(grads):
        if comm.Get_rank() == 0:
            token = mpi4jax.send(j, dest=((rank+1) % size), comm=comm, token=token)
            grads[(i+1) % size], token = mpi4jax.recv(j, source=((rank-1) % size), comm=comm, token=token)

        else:
            grads[(i+1) % size], token = mpi4jax.recv(j, source=((rank-1) % size), comm=comm, token=token)
            token = mpi4jax.send(j, dest=((rank+1) % size), comm=comm, token=token)

    grads = jnp.concatenate((grads[:(rank-2) % size], grads[(rank-2) % size:]))  # arrange back to norm
    grads = jax.vmap(lambda x: x/size)(grads) #this wont work just temp

    if not compression:
        return jax.tree_unflatten(flat_grads[1], grads)
    else:
        return grads













def PairAverageOpt(comm, params):
    pass


def update_params(optimizer_, grads, opt_state, params, comm,synchronicity=False, compression=True, compression_d_type="float16"):

    # PREPARE GRADIENTS
    if compression:
        compressed_grads, grads_d_type = grad_compress(grads, compression_d_type)

        # GET GRADIENTS
        if synchronicity:
            recieved_grads = ring_all_reduce(comm, compressed_grads)

        if not synchronicity:
            PairAverageOpt(comm, params)

        # HANDLE GRADIENTS
        grads = grad_uncompress(grads, rec)

    # PREPARE GRADIENTS
    else:
        pass
        # GET GRADIENTS

        # HANDLE GRADIENTS


    # FINAL UPDATE
    if synchronicity:
        updates, opt_state = optimizer_.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return params, opt_state
