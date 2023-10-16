import json
import numpy
from pathlib import Path


def get_label_saved_name(topic):
    """
    Some labels have a "/" in them, which is not allowed in a file name.
    This function replaces the "/" with " OR " to make it a valid file name.
    """
    return topic.replace("/", " OR ")


class MLJsonSerializer(json.JSONEncoder):
    """Custom encoder for json serialization of numpy.float32 and pathlib.Path objects."""

    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.float32):
            return float(obj)
        elif isinstance(obj, numpy.int_):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj.absolute())
        else:
            return super(MLJsonSerializer, self).default(obj)
