import tensorflow as tf
import tensorflow.keras as keras
import horovod.tensorflow as hvd_tf
import horovod.keras as hvd_keras
import horovod.torch as hvd_torch
import torch
import jax
from mpi4py import MPI
import arg_parser


#TODO find a way tto pass args to inits

def tensorflow_init():
    pass

def keras_init():
    pass

def horovod_tensorflow_init():
    hvd_tf.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd_tf.local_rank()], 'GPU')

def horovod_keras_init():
    hvd_keras.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd_keras.local_rank()], 'GPU')

def horovod_pytorch_init():
    hvd_torch.init()
    torch.cuda.set_device(hvd_torch.local_rank())


def pytorch_init():
    # TODO SET MASTER PORT/ADDR enviro vars
    args = arg_parser.Decint_ai_node_arg_parser().parse_args()
    torch.distributed.init_process_group(init_method=f"tcp://{args.my_address + ':' + args.my_port}",
                                         rank=args.my_rank, world_size=args.size)

def jax_init():
    # not necessary
    comm = MPI.COMM_WORLD
    return comm
