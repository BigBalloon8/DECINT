import copy
import quickle

def loads(serialized_obj, obj):
    un_serialized_attributes = quickle.loads(serialized_obj)
    obj = copy.copy(obj)
    for keys, vals in un_serialized_attributes.items():
        setattr(obj, keys, vals)
    return obj
