import mpi4jax
import jax
import jax.numpy as jnp
import threading
import DECINT_ai
import decint_jax
import multiprocessing
from jax_to_multipro import params_to_multiprocess, multiprocess_to_params


class PairAverageProccess(multiprocessing.Process):
    """
    wait for param request from another rank_i
    send params to another rank_i
    """

    def __init__(self, params, comm, compression=True, comp_dtype="float16"):
        multiprocessing.Process.__init__(self)
        self.params = params_to_multiprocess(params)
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.params_dtype = str(self.params[0].dtype)

    def run(self):
        while True:
            dest = DECINT_ai.node.message_handler("REQ")
            if dest:
                self._send_params(dest)

    @mpi4jax.experimental.auto_tokenize
    def _send_params(self, dest):
        if self.compression:
            jax.jit(mpi4jax.send)(decint_jax.grad_compress(self.params, self.comp_dtype), dest=dest, comm=self.comm)
        else:
            jax.jit(mpi4jax.send)(self.params, dest=dest, comm=self.comm)

    def __setitem__(self, key, value):
        if key == "params":
            for i in range(len(self.params)):
                self.params[i][:] = value[i][:]
        else:
            raise KeyError(f"{key} is not a valid key of PairAverageThread")

    def __getitem__(self, key):
        if key == "params":
            return multiprocess_to_params(self.params)
        else:
            raise KeyError


class DecintMpiThread(multiprocessing.Process):
    """
    constantly receive gradients from other nodes and average
    """

    def __init__(self, gradients, comm, compression=True, comp_dtype="float16"):
        multiprocessing.Process.__init__(self)
        self.gradients = params_to_multiprocess(self.zeros(gradients))
        self.comm = comm
        self.num_gradients = multiprocessing.Value("i", 0)
        self.token = None
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.gradients_dtype = str(self.gradients[0].dtype)
        self.zeros_grads = self.zeros(gradients)

    @staticmethod
    def zeros(gradients):
        return [jnp.zeros_like(i) for i in gradients]

    def run(self):
        while True:
            received_gradients = self._recv_grads()
            self.gradients = params_to_multiprocess(jax.tree_map(jnp.sum(), received_gradients, self.gradients))
            self.num_gradients += 1

    @mpi4jax.experimental.auto_tokenize
    def _recv_grads(self):
        if self.compression:
            received_gradients = mpi4jax.recv(decint_jax.grad_compress(self.gradients, self.comp_dtype),
                                              ((self.rank - 1) % self.size),
                                              comm=self.comm)
            received_gradients = decint_jax.grad_uncompress(self.gradients, received_gradients, self.gradients_dtype)
        else:
            received_gradients = mpi4jax.recv(self.gradients,
                                              ((self.rank - 1) % self.size),
                                              comm=self.comm)
        return received_gradients

    @mpi4jax.experimental.auto_tokenize
    def _send_grads(self, grads):
        if self.compression:
            self.token = mpi4jax.send(decint_jax.grad_compress(grads, self.comp_dtype),
                                      (self.rank + 1) % self.size, comm=self.comm, token=self.token)
        else:
            self.token = mpi4jax.send(grads, self.comp_dtype, (self.rank + 1) % self.size, comm=self.comm,
                                      token=self.token)

    def __getitem__(self, key):
        if key == "grads":
            if self.num_gradients == 0:
                raise Exception(
                    "must call put_grads before running getting params")  # TODO use custom Exception or skip this step
            ret_grads = jax.tree_map(lambda x: x / self.num_gradients, multiprocess_to_params(self.gradients))
            self.gradients = params_to_multiprocess(self.zeros_grads)
            self.num_gradients = 0
            return ret_grads
        else:
            raise KeyError(f"{key} is not a valid key of PairAverageThread")

    def put_grads(self, gradients_):
        self.gradients = params_to_multiprocess(jax.tree_map(jnp.sum(), gradients_, self.gradients))
        self.num_gradients += 1
        self._send_grads(gradients_)

    def __setitem__(self, key, value):
        if key == "grads":
            self.put_grads(value)
        else:
            raise KeyError(f"{key} is not a valid key of PairAverageThread")

