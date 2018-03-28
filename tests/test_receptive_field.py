"""Tests for receptive_fields module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tefla.core.layers import conv2d, max_pool
from tefla.core import receptive_field


def create_test_network_1():
  """Aligned network for test.

    The graph corresponds to the example from the second figure in
    go/cnn-rf-computation#arbitrary-computation-graphs

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = conv2d(l2_pad, 1, False, None, filter_size=3, stride=2, name='L2', padding='VALID')
    l3 = conv2d(l2, 1, False, None, filter_size=1, stride=2, name='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_2():
  """Aligned network for test.

    The graph corresponds to a variation to the example from the second figure in
    go/cnn-rf-computation#arbitrary-computation-graphs. Layers 2 and 3 are changed
    to max-pooling operations. Since the functionality is the same as convolution,
    the network is aligned and the receptive field size is the same as from the
    network created using create_test_network_1().

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = max_pool(l2_pad, filter_size=3, stride=2, name='L2', padding='VALID')
    l3 = max_pool(l2, filter_size=1, stride=2, name='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_3():
  """Misaligned network for test.

    The graph corresponds to the example from the first figure in
    go/cnn-rf-computation#arbitrary-computation-graphs

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1_pad = tf.pad(x, [[0, 0], [2, 1], [2, 1], [0, 0]])
    l1 = conv2d(l1_pad, 1, False, None, filter_size=5, stride=2, name='L1', padding='VALID')
    # Right branch.
    l2 = conv2d(x, 1, False, None, filter_size=3, stride=1, name='L2', padding='VALID')
    l3 = conv2d(l2, 1, False, None, filter_size=3, stride=1, name='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_4():
  """Misaligned network for test.

    The graph corresponds to a variation from the example from the second figure
    in go/cnn-rf-computation#arbitrary-computation-graphs. Layer 2 uses 'SAME'
    padding, which makes its padding dependent on the input image dimensionality.
    In this case, the effective padding will be undetermined, and the utility is
    not able to check the network alignment.

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    # Right branch.
    l2 = conv2d(x, 1, False, None, filter_size=3, stride=2, name='L2', padding='SAME')
    l3 = conv2d(l2, 1, False, None, filter_size=1, stride=2, name='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_5():
  """Single-path network for testing non-square kernels.

    The graph is similar to the right branch of the graph from
    create_test_network_1(), except that the kernel sizes are changed to be
    non-square.

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Two convolutional layers, where the first one has non-square kernel.
    l1 = conv2d(x, 1, False, None, filter_size=[3, 5], stride=2, name='L1', padding='VALID')
    l2 = conv2d(l1, 1, False, None, filter_size=[3, 1], stride=2, name='L2', padding='VALID')
    # ReLU.
    tf.nn.relu(l2, name='output')
  return g


def create_test_network_6():
  """Aligned network with dropout for test.

    The graph is similar to create_test_network_1(), except that the right branch
    has dropout normalization.

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = conv2d(l2_pad, 1, False, None, filter_size=3, stride=2, name='L2', padding='VALID')
    l3 = conv2d(l2, 1, False, None, filter_size=1, stride=2, name='L3', padding='VALID')
    dropout = tf.nn.dropout(l3, 0.5, name='dropout')
    # Addition.
    tf.nn.relu(l1 + dropout, name='output')
  return g


def create_test_network_7():
  """Aligned network for test, with a control dependency.

    The graph is similar to create_test_network_1(), except that it includes an
    assert operation on the left branch.

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = tf.placeholder(tf.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    l1_shape = tf.shape(l1)
    assert_op = tf.Assert(tf.equal(l1_shape[1], 2), [l1_shape], summarize=4)
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = conv2d(l2_pad, 1, False, None, filter_size=3, stride=2, name='L2', padding='VALID')
    l3 = conv2d(l2, 1, False, None, filter_size=1, stride=2, name='L3', padding='VALID')
    # Addition.
    with tf.control_dependencies([assert_op]):
      tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_8():
  """Aligned network for test, including an intermediate addition.

    The graph is similar to create_test_network_1(), except that it includes a few
    more layers on top. The added layers compose two different branches whose
    receptive fields are different. This makes this test case more challenging; in
    particular, this test fails if a naive DFS-like algorithm is used for RF
    computation.

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch before first addition.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='VALID')
    # Right branch before first addition.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = conv2d(l2_pad, 1, False, None, filter_size=3, stride=2, name='L2', padding='VALID')
    l3 = conv2d(l2, 1, False, None, filter_size=1, stride=2, name='L3', padding='VALID')
    # First addition.
    l4 = tf.nn.relu(l1 + l3)
    # Left branch after first addition.
    l5 = conv2d(l4, 1, False, None, filter_size=1, stride=2, name='L5', padding='VALID')
    # Right branch after first addition.
    l6_pad = tf.pad(l4, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l6 = conv2d(l6_pad, 1, False, None, filter_size=3, stride=2, name='L6', padding='VALID')
    # Final addition.
    tf.nn.relu(l5 + l6, name='output')

  return g


def create_test_network_9():
  """Aligned network for test, including an intermediate addition.

    The graph is the same as create_test_network_8(), except that VALID padding is
    changed to SAME.

    Returns:
      g: Tensorflow graph object (Graph proto).
    """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch before first addition.
    l1 = conv2d(x, 1, False, None, filter_size=1, stride=4, name='L1', padding='SAME')
    # Right branch before first addition.
    l2 = conv2d(x, 1, False, None, filter_size=3, stride=2, name='L2', padding='SAME')
    l3 = conv2d(l2, 1, False, None, filter_size=1, stride=2, name='L3', padding='SAME')
    # First addition.
    l4 = tf.nn.relu(l1 + l3)
    # Left branch after first addition.
    l5 = conv2d(l4, 1, False, None, filter_size=1, stride=2, name='L5', padding='SAME')
    # Right branch after first addition.
    l6 = conv2d(l4, 1, False, None, filter_size=3, stride=2, name='L6', padding='SAME')
    # Final addition.
    tf.nn.relu(l5 + l6, name='output')

  return g


class ReceptiveFieldTest(tf.test.TestCase):

  def testComputeRFFromGraphDefAligned(self):
    graph_def = create_test_network_1().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefAligned2(self):
    graph_def = create_test_network_2().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefUnaligned(self):
    graph_def = create_test_network_3().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    with self.assertRaises(ValueError):
      receptive_field.compute_receptive_field_from_graph_def(graph_def, input_node, output_node)

  def testComputeRFFromGraphDefUndefinedPadding(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, None)
    self.assertEqual(effective_padding_y, None)

  def testComputeRFFromGraphDefFixedInputDim(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node, input_resolution=[9, 9]))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefUnalignedFixedInputDim(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    with self.assertRaises(ValueError):
      receptive_field.compute_receptive_field_from_graph_def(
          graph_def, input_node, output_node, input_resolution=[8, 8])

  def testComputeRFFromGraphDefNonSquareRF(self):
    graph_def = create_test_network_5().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 5)
    self.assertEqual(receptive_field_y, 7)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 0)
    self.assertEqual(effective_padding_y, 0)

  def testComputeRFFromGraphDefStopPropagation(self):
    graph_def = create_test_network_6().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    # Compute the receptive field but stop the propagation for the random
    # uniform variable of the dropout.
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node, ['dropout/random_uniform']))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeCoordinatesRoundtrip(self):
    graph_def = create_test_network_1()
    input_node = 'input_image'
    output_node = 'output'
    rf = receptive_field.compute_receptive_field_from_graph_def(graph_def, input_node, output_node)

    x = np.random.randint(0, 100, (50, 2))
    y = rf.compute_feature_coordinates(x)
    x2 = rf.compute_input_center_coordinates(y)

    self.assertAllEqual(x, x2)

  def testComputeRFFromGraphDefAlignedWithControlDependencies(self):
    graph_def = create_test_network_7().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefWithIntermediateAddNode(self):
    graph_def = create_test_network_8().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 11)
    self.assertEqual(receptive_field_y, 11)
    self.assertEqual(effective_stride_x, 8)
    self.assertEqual(effective_stride_y, 8)
    self.assertEqual(effective_padding_x, 5)
    self.assertEqual(effective_padding_y, 5)

  def testComputeRFFromGraphDefWithIntermediateAddNodeSamePaddingFixedInputDim(self):
    graph_def = create_test_network_9().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
     effective_padding_x,
     effective_padding_y) = (receptive_field.compute_receptive_field_from_graph_def(
         graph_def, input_node, output_node, input_resolution=[17, 17]))
    self.assertEqual(receptive_field_x, 11)
    self.assertEqual(receptive_field_y, 11)
    self.assertEqual(effective_stride_x, 8)
    self.assertEqual(effective_stride_y, 8)
    self.assertEqual(effective_padding_x, 5)
    self.assertEqual(effective_padding_y, 5)


if __name__ == '__main__':
  tf.test.main()
