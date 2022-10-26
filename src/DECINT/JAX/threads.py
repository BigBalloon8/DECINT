import mpi4jax
import jax
import jax.numpy as jnp
import threading
import DECINT_ai
import decint_jax


class PairAverageThread(threading.Thread):
    def __init__(self, params, comm, compression=True, comp_dtype="float16"):
        super().__init__()
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
    def __init__(self, params, comm, compression=True, comp_dtype="float16"):
        # grads something the same shape as grads
        super().__init__()
        self.params = self.zeros(params)
        self.comm = comm
        self.num_params = 0
        self.token = None
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.params_dtype = str(self.params[0][0].dtype)

    @staticmethod
    def zeros(params):
        return [jnp.zeros_like(i) for i in params]

    def run(self):
        while True:
            if self.compression:
                recieved_params, self.token = mpi4jax.recv(decint_jax.grad_compress(self.params, self.comp_dtype),
                                                           ((self.rank - 1) % self.size),
                                                           comm=self.comm, token=self.token)
                recieved_params = decint_jax.grad_uncompress(self.params, recieved_params, self.params_dtype)
            else:
                recieved_params, self.token = mpi4jax.recv(self.params,
                                                           ((self.rank - 1) % self.size),
                                                           comm=self.comm, token=self.token)

            self.params = jax.tree_map(lambda x, y: jnp.sum(x, y), recieved_params, self.params)
            self.num_params += 1

    def __getitem__(self, key):
        if key == "params":
            if self.num_params == 0:
                raise Exception(
                    "must call put params before running func")  # TODO use custom Exception or just skip this step
            ret_params = jax.tree_map(lambda x: x / self.num_params, self.params)
            self.params = self.zeros(self.params)
            return ret_params

    def put_params(self, params_):
        self.params = jax.tree_map(lambda x, y: jnp.sum(x, y), params_, self.params)
        self.num_params += 1
        if self.compression:
            self.token = mpi4jax.send(decint_jax.grad_compress(params_, self.comp_dtype), (self.rank + 1) % self.size, comm=self.comm, token=self.token)
        else:
            self.token = mpi4jax.send(params_, self.comp_dtype, (self.rank + 1) % self.size, comm=self.comm, token=self.token)
