import mpi4jax
import jax
import jax.numpy as jnp
import threading
import DECINT_ai
import decint_jax


class PairAverageThread(threading.Thread):

    def __init__(self, params, comm, compression=True, comp_dtype="float16"):
        threading.Thread.__init__(self)
        self.params = params
        self.token = None
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.params_dtype = str(self.params[0][0].dtype)

    def run(self):
        while True:
            dest = DECINT_ai.node.message_handler("REQ")
            if dest:
                if self.compression:
                    self.token = jax.jit(mpi4jax.send)(decint_jax.grad_compress(self.params, self.comp_dtype), dest=dest, comm=self.comm, token=self.token)
                else:
                    self.token = jax.jit(mpi4jax.send)(self.params, dest=dest, comm=self.comm, token=self.token)

    def __setitem__(self, key, value):
        if key == "params":
            self.params = value
        else:
            raise KeyError

    def __getitem__(self, key):
        if key == "params":
            return self.params
        else:
            raise KeyError


class DecintMpiThread(threading.Thread):
    """
    constantly receive gradients from other nodes and average
    """
    def __int__(self, gradients, comm, compression=True, comp_dtype="float16"):
        threading.Thread.__init__(self)
        self.gradients = self.zeros(gradients)
        self.comm = comm
        self.num_gradients = 0
        self.token = None
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.gradients_dtype = str(self.gradients[0][0].dtype)
    @staticmethod
    def zeros(gradients):
        return [jnp.zeros_like(i) for i in gradients]

    def run(self):
        while True:
            if self.compression:
                received_gradients, self.token = mpi4jax.recv(decint_jax.grad_compress(self.gradients, self.comp_dtype),
                                                           ((self.rank - 1) % self.size),
                                                           comm=self.comm, token=self.token)
                received_gradients = decint_jax.grad_uncompress(self.gradients, received_gradients, self.gradients_dtype)
            else:
                received_gradients, self.token = mpi4jax.recv(self.gradients,
                                                           ((self.rank - 1) % self.size),
                                                           comm=self.comm, token=self.token)

            self.gradients = jax.tree_map(jnp.sum(), received_gradients, self.gradients)
            self.num_gradients += 1

    def __getitem__(self, key):
        if key == "grads":
            if self.num_gradients == 0:
                raise Exception(
                    "must call put_grads before running func")  # TODO use custom Exception or just skip this step
            ret_grads = jax.tree_map(lambda x: x / self.num_gradients, self.gradients)
            self.gradients = self.zeros(self.gradients)
            return ret_grads

    def put_grads(self, gradients_):
        self.gradients = jax.tree_map(jnp.sum(), gradients_, self.gradients)
        self.num_gradients += 1
        if self.compression:
            self.token = mpi4jax.send(decint_jax.grad_compress(gradients_, self.comp_dtype), (self.rank + 1) % self.size, comm=self.comm, token=self.token)
        else:
            self.token = mpi4jax.send(gradients_, self.comp_dtype, (self.rank + 1) % self.size, comm=self.comm, token=self.token)


