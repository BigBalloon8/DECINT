"""
Pickle Is a standard python library for object serelization and is used by mpi4py(horovod & mpi4jax)
to serialize object before communication.

The problem with this is the danger of recieving a potentially malicous object from a different node
so we have to create our on serelization method to and then build mpi4py from source
"""
from typing import Any
import quickle
import loads


def dumps(obj: Any) -> bytes:
    attribute_kws = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    attributes = {}
    for i in attribute_kws:
        attributes[i] = getattr(obj, i)
    return quickle.dumps(attributes)

if __name__ == "__main__":
    class test:
        def __init__(self):
            self.a = 1
            self.b = 7.5
            self.c = "hello"

        def test(self):
            print(self.a, self.b ,self.c)

    x = test()
    dumped = dumps(x)
    y = loads.loads(dumped, x)
    x.test()
    y.test()
