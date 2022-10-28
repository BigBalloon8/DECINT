import tensorflow as tf
import typing
from contextlib import suppress
from collections import Iterable

def model_wrapper(model, return_strategy=False):
    comm_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=comm_options)
    with strategy.scope():
        multi_worker_model = model
    if return_strategy:
        return multi_worker_model, strategy
    return model

def loss_and_opt_wrapper(strategy, loss_func, optimizer):
    #when running a custom training loop
    with strategy.scope():
        mirrored_opt, mirrored_loss = optimizer, loss_func
    return mirrored_loss, mirrored_opt

def ctl_train_step_wrapper(step_func: typing.Callable, stratagy: tf.distribute.Strategy, axis=None):
    @tf.function
    def decorated_step_func(*args, **kwargs):
        returned_statements = stratagy.run(step_func, args=args, kwargs=kwargs)  # TODO check this works
        statements = []
        if isinstance(returned_statements, Iterable):
            for i in returned_statements:
                statements.append(stratagy.reduce(tf.distribute.ReduceOp.SUM, i, axis=axis))
            return statements
        return stratagy.reduce(tf.distribute.ReduceOp.SUM, returned_statements, axis=axis)
    return decorated_step_func
