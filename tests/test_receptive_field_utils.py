"""Tests for receptive field module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

from tefla.core.layers import conv2d, max_pool
from tefla.core import receptive_field


def create_test_network():
  """Convolutional neural network for test.

    Returns:
      name_to_node: Dict keyed by node name, each entry containing the node's
        NodeDef.
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(dtypes.float32, (None, None, None, 1), name='input_image')
    # Left branch before first addition.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    # Right branch before first addition.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]], name='L2_pad')
    l2 = conv2d(l2_pad, 1, False, None, filter_size=3, stride=2, name='L2', padding='VALID')
    l3 = max_pool(l2, filter_size=3, stride=2, name='L3', padding='SAME')
    # First addition.
    l4 = tf.nn.relu(l1 + l3, name='L4_relu')
    # Left branch after first addition.
    l5 = conv2d(l4, 1, False, None, filter_size=1, stride=2, name='L5', padding='SAME')
    # Right branch after first addition.
    l6 = conv2d(l4, 1, False, None, filter_size=3, stride=2, name='L6', padding='SAME')
    # Final addition.
    tf.add(l5, l6, name='L7_add')

  name_to_node = receptive_field.parse_graph_nodes(g.as_graph_def())
  return g, name_to_node


class ParseLayerParametersTest(test.TestCase):

  def testParametersAreParsedCorrectly(self):
    """Checks parameters from create_test_network() are parsed correctly."""
    _, name_to_node = create_test_network()

    # L1.
    l1_node_name = 'L1/Conv2D'
    l1_params = receptive_field.get_layer_params(name_to_node[l1_node_name], name_to_node)
    expected_l1_params = (1, 1, 4, 4, 0, 0, 0, 0)
    self.assertEqual(l1_params, expected_l1_params)

    # L2 padding.
    l2_pad_name = 'L2_pad'
    l2_pad_params = receptive_field.get_layer_params(name_to_node[l2_pad_name], name_to_node)
    expected_l2_pad_params = (1, 1, 1, 1, 1, 1, 1, 1)
    self.assertEqual(l2_pad_params, expected_l2_pad_params)

    # L2.
    l2_node_name = 'L2/Conv2D'
    l2_params = receptive_field.get_layer_params(name_to_node[l2_node_name], name_to_node)
    expected_l2_params = (3, 3, 2, 2, 0, 0, 0, 0)
    self.assertEqual(l2_params, expected_l2_params)

    # L3.
    l3_node_name = 'L3/MaxPool'
    # - Without knowing input size.
    l3_params = receptive_field.get_layer_params(name_to_node[l3_node_name], name_to_node)
    expected_l3_params = (3, 3, 2, 2, None, None, None, None)
    self.assertEqual(l3_params, expected_l3_params)
    # - Input size is even.
    l3_even_params = receptive_field.get_layer_params(
        name_to_node[l3_node_name], name_to_node, input_resolution=[4, 4])
    expected_l3_even_params = (3, 3, 2, 2, 0, 0, 1, 1)
    self.assertEqual(l3_even_params, expected_l3_even_params)
    # - Input size is odd.
    l3_odd_params = receptive_field.get_layer_params(
        name_to_node[l3_node_name], name_to_node, input_resolution=[5, 5])
    expected_l3_odd_params = (3, 3, 2, 2, 1, 1, 2, 2)
    self.assertEqual(l3_odd_params, expected_l3_odd_params)

    # L4.
    l4_node_name = 'L4_relu'
    l4_params = receptive_field.get_layer_params(name_to_node[l4_node_name], name_to_node)
    expected_l4_params = (1, 1, 1, 1, 0, 0, 0, 0)
    self.assertEqual(l4_params, expected_l4_params)

    # L5.
    l5_node_name = 'L5/Conv2D'
    l5_params = receptive_field.get_layer_params(name_to_node[l5_node_name], name_to_node)
    expected_l5_params = (1, 1, 2, 2, 0, 0, 0, 0)
    self.assertEqual(l5_params, expected_l5_params)

    # L6.
    l6_node_name = 'L6/Conv2D'
    # - Without knowing input size.
    l6_params = receptive_field.get_layer_params(name_to_node[l6_node_name], name_to_node)
    expected_l6_params = (3, 3, 2, 2, None, None, None, None)
    self.assertEqual(l6_params, expected_l6_params)
    # - Input size is even.
    l6_even_params = receptive_field.get_layer_params(
        name_to_node[l6_node_name], name_to_node, input_resolution=[4, 4])
    expected_l6_even_params = (3, 3, 2, 2, 0, 0, 1, 1)
    self.assertEqual(l6_even_params, expected_l6_even_params)
    # - Input size is odd.
    l6_odd_params = receptive_field.get_layer_params(
        name_to_node[l6_node_name], name_to_node, input_resolution=[5, 5])
    expected_l6_odd_params = (3, 3, 2, 2, 1, 1, 2, 2)
    self.assertEqual(l6_odd_params, expected_l6_odd_params)

    # L7.
    l7_node_name = 'L7_add'
    l7_params = receptive_field.get_layer_params(name_to_node[l7_node_name], name_to_node)
    expected_l7_params = (1, 1, 1, 1, 0, 0, 0, 0)
    self.assertEqual(l7_params, expected_l7_params)


class GraphComputeOrderTest(test.TestCase):

  def check_topological_sort_and_sizes(self,
                                       node_info,
                                       expected_input_sizes=None,
                                       expected_output_sizes=None):
    """Helper function to check topological sorting and sizes are correct.

        The arguments expected_input_sizes and expected_output_sizes are used to
        check that the sizes are correct, if they are given.

        Args:
          node_info: Default dict keyed by node name, mapping to a named tuple with
            the following keys: {order, node, input_size, output_size}.
          expected_input_sizes: Dict mapping node names to expected input sizes
            (optional).
          expected_output_sizes: Dict mapping node names to expected output sizes
            (optional).
        """
    # Loop over nodes in sorted order, collecting those that were already seen.
    # These will be used to make sure that the graph is topologically sorted.
    # At the same time, we construct dicts from node name to input/output size,
    # which will be used to check those.
    already_seen_nodes = []
    input_sizes = {}
    output_sizes = {}
    for _, (_, node, input_size, output_size) in sorted(
        node_info.items(), key=lambda x: x[1].order):
      for inp_name in node.input:
        # Since the graph is topologically sorted, the inputs to the current
        # node must have been seen beforehand.
        self.assertIn(inp_name, already_seen_nodes)
      input_sizes[node.name] = input_size
      output_sizes[node.name] = output_size
      already_seen_nodes.append(node.name)

    # Check input sizes, if desired.
    if expected_input_sizes is not None:
      for k, v in expected_input_sizes.items():
        self.assertIn(k, input_sizes)
        self.assertEqual(input_sizes[k], v)

    # Check output sizes, if desired.
    if expected_output_sizes is not None:
      for k, v in expected_output_sizes.items():
        self.assertIn(k, output_sizes)
        self.assertEqual(output_sizes[k], v)

  def testGraphOrderIsCorrect(self):
    """Tests that the order and sizes of create_test_network() are correct."""

    graph_def = create_test_network()[0].as_graph_def()

    # Case 1: Input node name/size are not given.
    node_info, _ = receptive_field.get_compute_order(graph_def)
    self.check_topological_sort_and_sizes(node_info)

    # Case 2: Input node name is given, but not size.
    node_info, _ = receptive_field.get_compute_order(graph_def, input_node_name='input_image')
    self.check_topological_sort_and_sizes(node_info)

    # Case 3: Input node name and size (224) are given.
    node_info, _ = receptive_field.get_compute_order(
        graph_def, input_node_name='input_image', input_node_size=[224, 224])
    expected_input_sizes = {
        'input_image': None,
        'L1/Conv2D': [224, 224],
        'L2_pad': [224, 224],
        'L2/Conv2D': [225, 225],
        'L3/MaxPool': [112, 112],
        'L4_relu': [56, 56],
        'L5/Conv2D': [56, 56],
        'L6/Conv2D': [56, 56],
        'L7_add': [28, 28],
    }
    expected_output_sizes = {
        'input_image': [224, 224],
        'L1/Conv2D': [56, 56],
        'L2_pad': [225, 225],
        'L2/Conv2D': [112, 112],
        'L3/MaxPool': [56, 56],
        'L4_relu': [56, 56],
        'L5/Conv2D': [28, 28],
        'L6/Conv2D': [28, 28],
        'L7_add': [28, 28],
    }
    self.check_topological_sort_and_sizes(node_info, expected_input_sizes, expected_output_sizes)


if __name__ == '__main__':
  test.main()
