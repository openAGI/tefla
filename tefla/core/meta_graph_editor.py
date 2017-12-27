"""Apply graph_transforms tool to MetaGraphDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re as _re

import tensorflow as tf
from tensorflow.tools import graph_transforms as _graph_transforms


def _op_name(tensor_name):
    """Get the op name from a tensor name."""
    # control dependency inputs start with ^
    if tensor_name[0] == '^':
        tensor_name = tensor_name[1:]
    if ':' in tensor_name:
        op_name, _ = tensor_name.split(':')
        return op_name
    return tensor_name


def _get_shared_init_op(initializer_names):
    """Obtain the shared init op name, if it exists.

    Args:
     initializer_names: Dictionary of the "infrastructural" nodes (initializers,
       save and restore ops, etc.). The keys in this dictionary
       indicate the collection where these nodes were obtained from.

    Returns:
      A string indicating the shared init op name or none if None if none exists.
    """
    return_value = initializer_names.get(
        tf.saved_model.constants.MAIN_OP_KEY, None)
    if not return_value:
        return_value = initializer_names.get(
            tf.saved_model.constants.LEGACY_INIT_OP_KEY, None)
    return str(return_value[0]) if return_value else None


def _gtt_transforms(graph_def, input_names, output_names, initializer_names,
                    transforms):
    """Pass through gtt transforms, applying them to the graph_def.

    Args:
      graph_def: A GraphDef proto to be transformed.
      input_names: Names of input nodes.
      output_names: Names of output nodes.
      initializer_names: Dictionary of the "infrastructural" nodes (initializers,
        save and restore ops, etc.) that should be retained even if they are not
        transitively reachable from output nodes. The keys in this dictionary
        indicate the collection where these nodes were obtained from.
      transforms: A list of strings naming the graph transforms to be applied in
        order.
    Returns:
      The transformed GraphDef.
    """
    if not transforms:
        transformed_graph_def = tf.GraphDef()
        transformed_graph_def.CopyFrom(graph_def)
        return transformed_graph_def

    initializer_names_flat = sorted(
        [k for l in initializer_names.values() for k in l])
    all_output_names = output_names + initializer_names_flat
    return _graph_transforms.TransformGraph(graph_def, input_names,
                                            all_output_names, transforms)
