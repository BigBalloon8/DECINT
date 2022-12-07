from multiprocessing import shared_memory
import numpy as np
import jax.numpy as jnp


def params_to_multiprocess(params: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Converts a list of JAX arrays to a list of lists of shared memory numpy arrays.

    Args:
        params: A list of JAX arrays.

    Returns:
        A list of lists of sharedmemory numpy arrays.
    """
    s_m_params = []
    for i in params:
        shm = shared_memory.SharedMemory(create=True, size=i.nbytes)
        s_m_i = np.ndarray(i.shape, dtype=i.dtype, buffer=shm.buf)
        s_m_i[:] = i[:]
        s_m_params.append(s_m_i)
    return s_m_params


def multiprocess_to_params(s_m_params: list[np.ndarray]) -> list[list[jnp.ndarray]]:
    """Converts a list of lists of shared memory numpy arrays to a list of JAX arrays.

    Args:
        s_m_params: A list of lists of shared memory numpy arrays.

    Returns:
        A list of JAX arrays.
    """
    params = []
    for i in s_m_params:
        params.append(jnp.array(i))
    return params


