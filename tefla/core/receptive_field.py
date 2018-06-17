"""Functions to compute receptive field of a fully-convolutional network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import collections
import numpy as np
import tensorflow as tf

from tefla.core import logger as logging

# White-listed layer operations, which do not affect the receptive field
# computation.
_UNCHANGED_RF_LAYER_OPS = [
    "Add", "BiasAdd", "Cast", "Ceil", "ConcatV2", "Const", "Floor", "FusedBatchNorm", "Identity",
    "Log", "Mul", "Pow", "RealDiv", "Relu", "Relu6", "Round", "Rsqrt", "Softplus", "Sub",
    "VariableV2"
]

# Different ways in which padding modes may be spelled.
_VALID_PADDING = ["VALID", b"VALID"]
_SAME_PADDING = ["SAME", b"SAME"]


def _stride_size(node):
  """Computes stride size given a TF node.

  Args:
    node: Tensorflow node (NodeDef proto).

  Returns:
    stride_x: Stride size for horizontal direction (integer).
    stride_y: Stride size for vertical direction (integer).
  """
  strides_attr = node.attr["strides"]
  stride_y = strides_attr.list.i[1]
  stride_x = strides_attr.list.i[2]
  return stride_x, stride_y


def _conv_kernel_size(node, name_to_node):
  """Computes kernel size given a TF convolution or pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).

  Raises:
    ValueError: If the weight layer node is invalid.
  """
  weights_layer_read_name = node.input[1]
  if not weights_layer_read_name.endswith("/read"):
    raise ValueError("Weight layer's name input to conv layer does not end with '/read'")
  weights_layer_param_name = weights_layer_read_name[:-5]
  weights_node = name_to_node[weights_layer_param_name]
  if weights_node.op != "VariableV2":
    raise ValueError("Weight layer is not of type VariableV2")
  shape = weights_node.attr["shape"]
  kernel_size_y = shape.shape.dim[0].size
  kernel_size_x = shape.shape.dim[1].size
  return kernel_size_x, kernel_size_y


def _padding_size_conv_pool(node, kernel_size, stride, input_resolution=None):
  """Computes padding size given a TF convolution or pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).
    kernel_size: Kernel size of node (integer).
    stride: Stride size of node (integer).
    input_resolution: Input resolution to assume, if not None (integer).

  Returns:
    total_padding: Total padding size (integer).
    padding: Padding size, applied to the left or top (integer).

  Raises:
    ValueError: If padding is invalid.
  """
  # In this case, we need to carefully consider the different TF padding modes.
  # The padding depends on kernel size, and may depend on input size. If it
  # depends on input size and input_resolution is None, we raise an exception.
  padding_attr = node.attr["padding"]
  if padding_attr.s in _VALID_PADDING:
    total_padding = 0
    padding = 0
  elif padding_attr.s in _SAME_PADDING:
    if input_resolution is None:
      # In this case, we do not know the input resolution, so we can only know
      # the padding in some special cases.
      if kernel_size == 1:
        total_padding = 0
        padding = 0
      elif stride == 1:
        total_padding = kernel_size - 1
        padding = int(math.floor(float(total_padding) / 2))
      elif stride == 2 and kernel_size % 2 == 0:
        # In this case, we can be sure of the left/top padding, but not of the
        # total padding.
        total_padding = None
        padding = int(math.floor((float(kernel_size) - 1) / 2))
      else:
        total_padding = None
        padding = None
    else:
      # First, compute total_padding based on documentation.
      if input_resolution % stride == 0:
        total_padding = int(max(float(kernel_size - stride), 0.0))
      else:
        total_padding = int(max(float(kernel_size - (input_resolution % stride)), 0.0))
      # Then, compute left/top padding.
      padding = int(math.floor(float(total_padding) / 2))

  else:
    raise ValueError("Invalid padding operation %s" % padding_attr.s)
  return total_padding, padding


def _pool_kernel_size(node):
  """Computes kernel size given a TF pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).

  Raises:
    ValueError: If pooling is invalid.
  """
  ksize = node.attr["ksize"]
  kernel_size_y = ksize.list.i[1]
  kernel_size_x = ksize.list.i[2]
  if ksize.list.i[0] != 1:
    raise ValueError("pool ksize for first dim is not 1")
  if ksize.list.i[3] != 1:
    raise ValueError("pool ksize for last dim is not 1")
  return kernel_size_x, kernel_size_y


def _padding_size_pad_layer(node, name_to_node):
  """Computes padding size given a TF padding node.

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.

  Returns:
    total_padding_x: Total padding size for horizontal direction (integer).
    padding_x: Padding size for horizontal direction, left side (integer).
    total_padding_y: Total padding size for vertical direction (integer).
    padding_y: Padding size for vertical direction, top side (integer).

  Raises:
    ValueError: If padding layer is invalid.
  """
  paddings_layer_name = node.input[1]
  if not paddings_layer_name.endswith("/paddings"):
    raise ValueError("Padding layer name does not end with '/paddings'")
  paddings_node = name_to_node[paddings_layer_name]
  if paddings_node.op != "Const":
    raise ValueError("Padding op is not Const")
  value = paddings_node.attr["value"]
  t = tf.make_ndarray(value.tensor)
  padding_y = t[1][0]
  padding_x = t[2][0]
  total_padding_y = padding_y + t[1][1]
  total_padding_x = padding_x + t[2][1]
  if (t[0][0] != 0) or (t[0][1] != 0):
    raise ValueError("padding is not zero for first tensor dim")
  if (t[3][0] != 0) or (t[3][1] != 0):
    raise ValueError("padding is not zero for last tensor dim")
  return total_padding_x, padding_x, total_padding_y, padding_y


def get_layer_params(node, name_to_node, input_resolution=None, force=False):
  """Gets layer parameters relevant for RF computation.

  Currently, only these nodes are supported:
  - Conv2D
  - DepthwiseConv2dNative
  - Pad
  - MaxPool
  - AvgPool
  - all nodes listed in _UNCHANGED_RF_LAYER_OPS

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
    input_resolution: List with 2 dimensions, denoting the height/width of the
      input feature map to this layer. If set to None, then the padding may be
      undefined (in tensorflow, SAME padding depends on input spatial
      resolution).
    force: If True, the function does not raise a ValueError if the layer op is
      unknown. Instead, in this case it sets each of the returned parameters to
      None.

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).
    stride_x: Stride size for horizontal direction (integer).
    stride_y: Stride size for vertical direction (integer).
    padding_x: Padding size for horizontal direction, left side (integer).
    padding_y: Padding size for vertical direction, top side (integer).
    total_padding_x: Total padding size for horizontal direction (integer).
    total_padding_y: Total padding size for vertical direction (integer).

  Raises:
    ValueError: If layer op is unknown and force is False.
  """
  if node.op == "Conv2D" or node.op == "DepthwiseConv2dNative":
    stride_x, stride_y = _stride_size(node)
    kernel_size_x, kernel_size_y = _conv_kernel_size(node, name_to_node)
    # Compute the padding for this node separately for each direction.
    total_padding_x, padding_x = _padding_size_conv_pool(node, kernel_size_x, stride_x,
                                                         input_resolution[1]
                                                         if input_resolution is not None else None)
    total_padding_y, padding_y = _padding_size_conv_pool(node, kernel_size_y, stride_y,
                                                         input_resolution[0]
                                                         if input_resolution is not None else None)
  elif node.op == "Pad":
    # Kernel and stride are simply 1 in this case.
    kernel_size_x = 1
    kernel_size_y = 1
    stride_x = 1
    stride_y = 1
    total_padding_x, padding_x, total_padding_y, padding_y = (_padding_size_pad_layer(
        node, name_to_node))
  elif node.op == "MaxPool" or node.op == "AvgPool":
    stride_x, stride_y = _stride_size(node)
    kernel_size_x, kernel_size_y = _pool_kernel_size(node)
    # Compute the padding for this node separately for each direction.
    total_padding_x, padding_x = _padding_size_conv_pool(node, kernel_size_x, stride_x,
                                                         input_resolution[1]
                                                         if input_resolution is not None else None)
    total_padding_y, padding_y = _padding_size_conv_pool(node, kernel_size_y, stride_y,
                                                         input_resolution[0]
                                                         if input_resolution is not None else None)
  elif node.op in _UNCHANGED_RF_LAYER_OPS:
    # These nodes do not modify the RF parameters.
    kernel_size_x = 1
    kernel_size_y = 1
    stride_x = 1
    stride_y = 1
    total_padding_x = 0
    padding_x = 0
    total_padding_y = 0
    padding_y = 0
  else:
    if force:
      kernel_size_x = None
      kernel_size_y = None
      stride_x = None
      stride_y = None
      total_padding_x = None
      padding_x = None
      total_padding_y = None
      padding_y = None
    else:
      raise ValueError("Unknown layer for operation '%s': %s" % (node.name, node.op))
  return (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, padding_y, total_padding_x,
          total_padding_y)


def parse_graph_nodes(graph_def):
  """Helper function to parse GraphDef's nodes.

  It returns a dict mapping from node name to NodeDef.

  Args:
    graph_def: A GraphDef object.

  Returns:
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
  """
  name_to_node = {}
  for node_def in graph_def.node:
    name_to_node[node_def.name] = node_def
  return name_to_node


# Named tuple used to collect information from each node in a computation graph.
_node_info = collections.namedtuple(
    'NodeInfo', field_names=['order', 'node', 'input_size', 'output_size'])


def _compute_output_resolution(input_spatial_resolution, kernel_size, stride, total_padding):
  """Computes output resolution, given input resolution and layer parameters.

  Note that this computation is done only over one dimension (eg, x or y).
  If any of the inputs is None, returns None.

  Args:
    input_spatial_resolution: Input spatial resolution (int).
    kernel_size: Kernel size (int).
    stride: Stride (int).
    total_padding: Total padding to be applied (int).
  Returns:
    output_resolution: Ouput dimension (int) or None.
  """
  if (input_spatial_resolution is None) or (kernel_size is
                                            None) or (stride is None) or (total_padding is None):
    return None
  return int(math.ceil((input_spatial_resolution + total_padding - kernel_size + 1) / stride))


def _get_computed_nodes(name_to_node, current, node_info, input_node_name='', input_node_size=None):
  """Traverses the graph recursively to compute its topological order.

  Optionally, the function may also compute the input and output feature map
  resolutions at each node. In this case, input_node_name and input_node_size
  must be set. Note that if a node's op type is unknown, the input and output
  resolutions are ignored and set to None.

  Args:
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
    current: Current node name.
    node_info: Map of nodes we've already traversed, containing their _node_info
      information.
    input_node_name: Name of node with fixed input resolution (optional).
    input_node_size: Fixed input resolution to use (optional).
  Returns:
    order: Order in topological sort for 'current'.
    input_size: Tensor spatial resolution at input of current node.
    output_size: Tensor spatial resolution at output of current node.
  """
  if current in node_info:
    return (node_info[current].order, node_info[current].input_size, node_info[current].output_size)

  node_def = name_to_node[current]

  if current == input_node_name:
    order = 0
    input_size = None
    output_size = input_node_size
    node_info[current] = _node_info(order, node_def, input_size, output_size)
    return (order, input_size, output_size)

  input_size = None
  output_size = None

  order = 0
  number_inputs = 0
  for each in node_def.input:
    # Parses name of input node.
    if each.startswith('^'):
      # The character '^' denotes a control dependency, so this input node can
      # be safely ignored.
      continue
    each = each.split(':')[0]
    # Recursively computes ordering.
    (parent_order, _, parent_output_size) = _get_computed_nodes(name_to_node, each, node_info,
                                                                input_node_name, input_node_size)
    order = max(order, parent_order + 1)
    if number_inputs == 0:
      # For all the types of nodes we consider, the first input corresponds to
      # the feature map.
      input_size = parent_output_size
    number_inputs += 1

  # Figure out output size for this layer.
  if input_size is None:
    output_size = None
  else:
    (kernel_size_x, kernel_size_y, stride_x, stride_y, _, _, total_padding_x,
     total_padding_y) = (get_layer_params(node_def, name_to_node, input_size, force=True))
    output_size = [None] * 2
    output_size[0] = _compute_output_resolution(input_size[0], kernel_size_x, stride_x,
                                                total_padding_x)
    output_size[1] = _compute_output_resolution(input_size[1], kernel_size_y, stride_y,
                                                total_padding_y)

  node_info[current] = _node_info(order, node_def, input_size, output_size)

  return order, input_size, output_size


def get_compute_order(graph_def, input_node_name='', input_node_size=None):
  """Computes order of computation for a given CNN graph.

  Optionally, the function may also compute the input and output feature map
  resolutions at each node. In this case, input_node_name and input_node_size
  must be set. Note that if a node's op type is unknown, the input and output
  resolutions are ignored and set to None.

  Args:
    graph_def: GraphDef object.
    input_node_name: Name of node with fixed input resolution (optional). This
      is usually the node name for the input image in a CNN.
    input_node_size: 2D list of integers, fixed input resolution to use
      (optional). This is usually the input resolution used for the input image
      in a CNN (common examples are: [224, 224], [299, 299], [321, 321]).
  Returns:
    node_info: Default dict keyed by node name, mapping to a named tuple with
      the following fields:
      - order: Integer denoting topological order;
      - node: NodeDef for the given node;
      - input_size: 2D list of integers, denoting the input spatial resolution
        to the node;
      - output_size: 2D list of integers, denoting the output spatial resolution
        of the node.
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
  """
  name_to_node = parse_graph_nodes(graph_def)
  node_info = collections.defaultdict(_node_info)
  for each in graph_def.node:
    _get_computed_nodes(name_to_node, each.name, node_info, input_node_name, input_node_size)
  return node_info, name_to_node


def _get_rf_size_node_input(stride, kernel_size, rf_size_output):
  """Computes RF size at the input of a given layer.

  Args:
    stride: Stride of given layer (integer).
    kernel_size: Kernel size of given layer (integer).
    rf_size_output: RF size at output of given layer (integer).

  Returns:
    rf_size_input: RF size at input of given layer (integer).
  """
  return stride * rf_size_output + kernel_size - stride


def _get_effective_stride_node_input(stride, effective_stride_output):
  """Computes effective stride at the input of a given layer.

  Args:
    stride: Stride of given layer (integer).
    effective_stride_output: Effective stride at output of given layer
      (integer).

  Returns:
    effective_stride_input: Effective stride at input of given layer
      (integer).
  """
  return stride * effective_stride_output


def _get_effective_padding_node_input(stride, padding, effective_padding_output):
  """Computes effective padding at the input of a given layer.

  Args:
    stride: Stride of given layer (integer).
    padding: Padding of given layer (integer).
    effective_padding_output: Effective padding at output of given layer
      (integer).

  Returns:
    effective_padding_input: Effective padding at input of given layer
      (integer).
  """
  return stride * effective_padding_output + padding


class ReceptiveField(object):
  """Receptive field of a convolutional neural network.

  Args:
    size: Receptive field size.
    stride: Effective stride.
    padding: Effective padding.
  """

  def __init__(self, size, stride, padding):
    self.size = np.asarray(size)
    self.stride = np.asarray(stride)
    self.padding = np.asarray(padding)

  def compute_input_center_coordinates(self, y, axis=None):
    """Computes the center of the receptive field that generated a feature.

    Args:
      y: An array of feature coordinates with shape `(..., d)`, where `d` is the
        number of dimensions of the coordinates.
      axis: The dimensions for which to compute the input center coordinates.
        If `None` (the default), compute the input center coordinates for all
        dimensions.

    Returns:
      x: Center of the receptive field that generated the features, at the input
        of the network.

    Raises:
      ValueError: If the number of dimensions of the feature coordinates does
        not match the number of elements in `axis`.
    """
    # Use all dimensions.
    if axis is None:
      axis = range(self.size.size)
    # Ensure axis is a list because tuples have different indexing behavior.
    axis = list(axis)
    y = np.asarray(y)
    if y.shape[-1] != len(axis):
      raise ValueError("Dimensionality of the feature coordinates `y` (%d) "
                       "does not match dimensionality of `axis` (%d)" % (y.shape[-1], len(axis)))
    return -self.padding[axis] + y * self.stride[axis] + (self.size[axis] - 1) / 2

  def compute_feature_coordinates(self, x, axis=None):
    """Computes the position of a feature given the center of a receptive
    field.

    Args:
      x: An array of input center coordinates with shape `(..., d)`, where `d`
        is the number of dimensions of the coordinates.
      axis: The dimensions for which to compute the feature coordinates.
        If `None` (the default), compute the feature coordinates for all
        dimensions.

    Returns:
      y: Coordinates of the features.

    Raises:
      ValueError: If the number of dimensions of the input center coordinates
        does not match the number of elements in `axis`.
    """
    # Use all dimensions.
    if axis is None:
      axis = range(self.size.size)
    # Ensure axis is a list because tuples have different indexing behavior.
    axis = list(axis)
    x = np.asarray(x)
    if x.shape[-1] != len(axis):
      raise ValueError("Dimensionality of the input center coordinates `x` "
                       "(%d) does not match dimensionality of `axis` (%d)" % (x.shape[-1],
                                                                              len(axis)))
    return (x + self.padding[axis] + (1 - self.size[axis]) / 2) / self.stride[axis]

  def __iter__(self):
    return iter(np.concatenate([self.size, self.stride, self.padding]))


def compute_receptive_field_from_graph_def(graph_def,
                                           input_node,
                                           output_node,
                                           stop_propagation=None,
                                           input_resolution=None):
  """Computes receptive field (RF) parameters from a Graph or GraphDef object.

  The algorithm stops the calculation of the receptive field whenever it
  encounters an operation in the list `stop_propagation`. Stopping the
  calculation early can be useful to calculate the receptive field of a
  subgraph such as a single branch of the
  [inception network](https://arxiv.org/abs/1512.00567).

  Args:
    graph_def: Graph or GraphDef object.
    input_node: Name of the input node or Tensor object from graph.
    output_node: Name of the output node or Tensor object from graph.
    stop_propagation: List of operations or scope names for which to stop the
      propagation of the receptive field.
    input_resolution: 2D list. If the input resolution to the model is fixed and
      known, this may be set. This is helpful for cases where the RF parameters
      vary depending on the input resolution (this happens since SAME padding in
      tensorflow depends on input resolution in general). If this is None, it is
      assumed that the input resolution is unknown, so some RF parameters may be
      unknown (depending on the model architecture).

  Returns:
    rf_size_x: Receptive field size of network in the horizontal direction, with
      respect to specified input and output.
    rf_size_y: Receptive field size of network in the vertical direction, with
      respect to specified input and output.
    effective_stride_x: Effective stride of network in the horizontal direction,
      with respect to specified input and output.
    effective_stride_y: Effective stride of network in the vertical direction,
      with respect to specified input and output.
    effective_padding_x: Effective padding of network in the horizontal
      direction, with respect to specified input and output.
    effective_padding_y: Effective padding of network in the vertical
      direction, with respect to specified input and output.

  Raises:
    ValueError: If network is not aligned or if either input or output nodes
      cannot be found. For network criterion alignment, see
      photos/vision/features/delf/g3doc/rf_computation.md
  """
  # Convert a graph to graph_def if necessary.
  if isinstance(graph_def, tf.Graph):
    graph_def = graph_def.as_graph_def()

  # Convert tensors to names.
  if isinstance(input_node, tf.Tensor):
    input_node = input_node.op.name
  if isinstance(output_node, tf.Tensor):
    output_node = output_node.op.name

  stop_propagation = stop_propagation or []

  # Computes order of computation for a given graph.
  node_info, name_to_node = get_compute_order(
      graph_def=graph_def, input_node_name=input_node, input_node_size=input_resolution)

  # Sort in reverse topological order.
  ordered_node_info = sorted(node_info.items(), key=lambda x: -x[1].order)

  # Dictionaries to keep track of receptive field, effective stride and
  # effective padding of different nodes.
  rf_sizes_x = {}
  rf_sizes_y = {}
  effective_strides_x = {}
  effective_strides_y = {}
  effective_paddings_x = {}
  effective_paddings_y = {}

  # Initialize dicts for output_node.
  rf_sizes_x[output_node] = 1
  rf_sizes_y[output_node] = 1
  effective_strides_x[output_node] = 1
  effective_strides_y[output_node] = 1
  effective_paddings_x[output_node] = 0
  effective_paddings_y[output_node] = 0

  # Flag to denote if we found output node yet. If we have not, we skip nodes
  # until the output node is found.
  found_output_node = False

  # Flag to denote if padding is undefined. This happens when SAME padding mode
  # is used in conjunction with stride and kernel sizes which make it such that
  # the padding to be applied would depend on the input size. In this case,
  # alignment checks are skipped, and the effective padding is None.
  undefined_padding = False

  for _, (o, node, _, _) in ordered_node_info:
    if node:
      logging.info("%10d %-100s %-20s" % (o, node.name[:90], node.op))
    else:
      continue

    # When we find input node, we can stop.
    if node.name == input_node:
      break

    # Loop until we find the output node. All nodes before finding the output
    # one are irrelevant, so they can be skipped.
    if not found_output_node:
      if node.name == output_node:
        found_output_node = True

    if found_output_node:
      if node.name not in rf_sizes_x:
        assert node.name not in rf_sizes_y, ("Node %s is in rf_sizes_y, but "
                                             "not in rf_sizes_x" % node.name)
        # In this case, node is not relevant since it's not part of the
        # computation we're interested in.
        logging.info("Irrelevant node %s, skipping it..." % node.name)
        continue

      # Get params for this layer.
      (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, padding_y, _,
       _) = get_layer_params(node, name_to_node, node_info[node.name].input_size)
      logging.info("kernel_size_x = %s, kernel_size_y = %s, "
                   "stride_x = %s, stride_y = %s, "
                   "padding_x = %s, padding_y = %s, input size = %s" %
                   (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, padding_y,
                    node_info[node.name].input_size))
      if padding_x is None or padding_y is None:
        undefined_padding = True

      # Get parameters at input of this layer which may or may not be propagated
      # to the input layers.
      rf_size_input_x = _get_rf_size_node_input(stride_x, kernel_size_x, rf_sizes_x[node.name])
      rf_size_input_y = _get_rf_size_node_input(stride_y, kernel_size_y, rf_sizes_y[node.name])
      effective_stride_input_x = _get_effective_stride_node_input(stride_x,
                                                                  effective_strides_x[node.name])
      effective_stride_input_y = _get_effective_stride_node_input(stride_y,
                                                                  effective_strides_y[node.name])
      if not undefined_padding:
        effective_padding_input_x = _get_effective_padding_node_input(
            stride_x, padding_x, effective_paddings_x[node.name])
        effective_padding_input_y = _get_effective_padding_node_input(
            stride_y, padding_y, effective_paddings_y[node.name])
      else:
        effective_padding_input_x = None
        effective_padding_input_y = None
      logging.info("rf_size_input_x = %s, rf_size_input_y = %s, "
                   "effective_stride_input_x = %s, effective_stride_input_y = %s, "
                   "effective_padding_input_x = %s, effective_padding_input_y = %s" %
                   (rf_size_input_x, rf_size_input_y, effective_stride_input_x,
                    effective_stride_input_y, effective_padding_input_x, effective_padding_input_y))

      # Loop over this node's inputs and potentially propagate information down.
      for inp_name in node.input:
        # Stop the propagation of the receptive field.
        if any(inp_name.startswith(stop) for stop in stop_propagation):
          logging.info("Skipping explicitly ignored node %s." % inp_name)
          continue

        logging.info("inp_name = %s" % inp_name)
        if inp_name.startswith("^"):
          # The character "^" denotes a control dependency, so this input node
          # can be safely ignored.
          continue

        inp_node = name_to_node[inp_name]
        logging.info("inp_node = \n%s" % inp_node)
        if inp_name in rf_sizes_x:
          assert inp_name in rf_sizes_y, ("Node %s is in rf_sizes_x, but "
                                          "not in rf_sizes_y" % inp_name)
          logging.info("rf_sizes_x[inp_name] = %s,"
                       " rf_sizes_y[inp_name] = %s, "
                       "effective_strides_x[inp_name] = %s,"
                       " effective_strides_y[inp_name] = %s, "
                       "effective_paddings_x[inp_name] = %s,"
                       " effective_paddings_y[inp_name] = %s" %
                       (rf_sizes_x[inp_name], rf_sizes_y[inp_name], effective_strides_x[inp_name],
                        effective_strides_y[inp_name], effective_paddings_x[inp_name],
                        effective_paddings_y[inp_name]))
          # This node was already discovered through a previous path, so we need
          # to make sure that graph is aligned. This alignment check is skipped
          # if the padding is not defined, since in this case alignment cannot
          # be checked.
          if not undefined_padding:
            if effective_strides_x[inp_name] != effective_stride_input_x:
              raise ValueError("Graph is not aligned since effective stride from different "
                               "paths is different in horizontal direction")
            if effective_strides_y[inp_name] != effective_stride_input_y:
              raise ValueError("Graph is not aligned since effective stride from different "
                               "paths is different in vertical direction")
            if (rf_sizes_x[inp_name] - 1) / 2 - effective_paddings_x[inp_name] != (
                    rf_size_input_x - 1) / 2 - effective_padding_input_x:
              raise ValueError("Graph is not aligned since center shift from different "
                               "paths is different in horizontal direction")
            if (rf_sizes_y[inp_name] - 1) / 2 - effective_paddings_y[inp_name] != (
                    rf_size_input_y - 1) / 2 - effective_padding_input_y:
              raise ValueError("Graph is not aligned since center shift from different "
                               "paths is different in vertical direction")
          # Keep track of path with largest RF, for both directions.
          if rf_sizes_x[inp_name] < rf_size_input_x:
            rf_sizes_x[inp_name] = rf_size_input_x
            effective_strides_x[inp_name] = effective_stride_input_x
            effective_paddings_x[inp_name] = effective_padding_input_x
          if rf_sizes_y[inp_name] < rf_size_input_y:
            rf_sizes_y[inp_name] = rf_size_input_y
            effective_strides_y[inp_name] = effective_stride_input_y
            effective_paddings_y[inp_name] = effective_padding_input_y
        else:
          assert inp_name not in rf_sizes_y, ("Node %s is in rf_sizes_y, but "
                                              "not in rf_sizes_x" % inp_name)
          # In this case, it is the first time we encounter this node. So we
          # propagate the RF parameters.
          rf_sizes_x[inp_name] = rf_size_input_x
          rf_sizes_y[inp_name] = rf_size_input_y
          effective_strides_x[inp_name] = effective_stride_input_x
          effective_strides_y[inp_name] = effective_stride_input_y
          effective_paddings_x[inp_name] = effective_padding_input_x
          effective_paddings_y[inp_name] = effective_padding_input_y

  if not found_output_node:
    raise ValueError("Output node was not found")
  if input_node not in rf_sizes_x:
    raise ValueError("Input node was not found")
  return ReceptiveField((rf_sizes_x[input_node], rf_sizes_y[input_node]),
                        (effective_strides_x[input_node], effective_strides_y[input_node]),
                        (effective_paddings_x[input_node], effective_paddings_y[input_node]))
