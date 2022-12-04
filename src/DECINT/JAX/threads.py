import mpi4jax
import jax
import jax.numpy as jnp
import threading
import DECINT_ai
import decint_jax
import multiprocessing



class PairAverageProccess(multiprocessing.Process):
    """
    wait for param request from another rank_i
    send params to another rank_i
    """

    def __init__(self, params, comm, compression=True, comp_dtype="float16"):
        multiprocessing.Process.__init__(self)
        self.param_queue = multiprocessing.Queue(maxsize=1)
        self.token_queue = multiprocessing.Queue(maxsize=1)
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.params_dtype = str(self.params[0][0].dtype)
        self.param_queue.put(params)

    def run(self):
        while True:
            dest = DECINT_ai.node.message_handler("REQ")
            if dest:
                params = self.param_queue.get()
                self.param_queue.put(params)
                if self.compression:
                    self.token_queue.put(jax.jit(mpi4jax.send)(decint_jax.grad_compress(params, self.comp_dtype), dest=dest, comm=self.comm, token=self.get_token()))
                else:
                    self.token_queue.put(jax.jit(mpi4jax.send)(params, dest=dest, comm=self.comm, token=self.get_token()))

    def __setitem__(self, key, value):
        if key == "params":
            self.param_queue.get()
            self.param_queue.put(value)
        else:
            raise KeyError(f"{key} is not a valid key of PairAverageThread")

    def __getitem__(self, key):
        if key == "params":
            params = self.param_queue.get()
            self.param_queue.put(params)
            return params
        else:
            raise KeyError

    def get_token(self):
        if self.token_queue.empty():
            return None
        else:
            return self.token_queue.get()


class DecintMpiThread(multiprocessing.Process):
    """
    constantly receive gradients from other nodes and average
    """
    def __int__(self, gradients, comm, compression=True, comp_dtype="float16"):
        multiprocessing.Process.__init__(self)
        self.gradient_queue = multiprocessing.Queue(maxsize=1)
        self.comm = comm
        self.num_gradients = multiprocessing.shared_memory.SharableList([0])
        self.token = None
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.compression = compression
        self.comp_dtype = comp_dtype
        self.gradients_dtype = str(self.gradients[0][0].dtype)
        self.zeros_grads = self.zeros(gradients)
        self.gradient_queue.put(self.zeros)
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
            self.num_gradients[0] = self.num_gradients[0] + 1

    def __getitem__(self, key):
        if key == "grads":
            if self.num_gradients[0] == 0:
                raise Exception(
                    "must call put_grads before running func")  # TODO use custom Exception or just skip this step
            ret_grads = jax.tree_map(lambda x: x / self.num_gradients[0], self.gradient_queue.get())
            self.gradient_queue.put(self.zeros_grads)
            return ret_grads

    def put_grads(self, gradients_):
        self.gradient_queue.put(jax.tree_map(jnp.sum(), gradients_, self.gradient_queue.get()))
        self.num_gradients[0] = self.num_gradients[0] + 1
        if self.compression:
            self.token = mpi4jax.send(decint_jax.grad_compress(gradients_, self.comp_dtype), (self.rank + 1) % self.size, comm=self.comm, token=self.token)
        else:
            self.token = mpi4jax.send(gradients_, self.comp_dtype, (self.rank + 1) % self.size, comm=self.comm, token=self.token)


