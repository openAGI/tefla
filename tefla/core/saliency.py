from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import numpy as np
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class SaliencyMask(object):
    """Base class for saliency masks. Alone, this class doesn't do anything."""

    def __init__(self, graph, session, y, x):
        """Constructs a SaliencyMask.

        Args:
            graph: The TensorFlow graph to evaluate masks on.
            session: The current TensorFlow session.
            y: The output tensor to compute the SaliencyMask against. This tensor
                should be of size 1.
            x: The input tensor to compute the SaliencyMask against. The outer
                dimension should be the batch size.
        """

        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys.
        size = 1
        for shape in y.shape:
            size *= shape
        assert size == 1

        self.graph = graph
        self.session = session
        self.y = y
        self.x = x

    @abc.abstractmethod
    def GetMask(self, x_value, feed_dict={}):
        """Returns an unsmoothed mask.

        Args:
            x_value: Input value, not batched.
            feed_dict: (Optional) feed dictionary to pass to the session.run call.

        Returns:
            returns a 3D mask
        """
        pass

    def GetSmoothedMask(
            self, x_value, feed_dict={}, stdev_spread=.2, nsamples=50):
        """Returns a mask that is smoothed with the SmoothGrad method.

        Args:
            x_value: Input value, not batched.
            feed_dict: (Optional) feed dictionary to pass to the session.run call.
        """
        stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

        total_gradients = np.zeros_like(x_value)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, x_value.shape)
            x_plus_noise = x_value + noise

            total_gradients += self.GetMask(x_plus_noise, feed_dict)

        return total_gradients / nsamples


class GradientSaliency(SaliencyMask):
    r"""A SaliencyMask class that computes saliency masks with a gradient."""

    def __init__(self, graph, session, y, x):
        super(GradientSaliency, self).__init__(graph, session, y, x)
        self.gradients_node = tf.gradients(y, x)[0]

    def GetMask(self, x_value, feed_dict={}):
        """Returns a vanilla gradient mask.

        Args:
            x_value: Input value, not batched.
            feed_dict: (Optional)feed dictionary to pass to the session.run call.
        """
        feed_dict[self.x] = [x_value]
        return self.session.run(self.gradients_node, feed_dict=feed_dict)[0]


class GuidedBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with GuidedBackProp.
    This implementation copies the TensorFlow graph to a new graph with the ReLU
    gradient overwritten as in the paper:
    https://arxiv.org/abs/1412.6806
    """

    GuidedReluRegistered = False

    def __init__(self, graph, session, y, x):
        """Constructs a GuidedBackprop SaliencyMask."""
        super(GuidedBackprop, self).__init__(graph, session, y, x)

        self.x = x

        if GuidedBackprop.GuidedReluRegistered is False:
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad

        GuidedBackprop.GuidedReluRegistered = True

        with graph.as_default():
            saver = tf.train.Saver()
            saver.save(session, '/tmp/guided_backprop_ckpt')

        graph_def = graph.as_graph_def()

        self.guided_graph = tf.Graph()
        with self.guided_graph.as_default():
            self.guided_sess = tf.Session(graph=self.guided_graph)
            with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                tf.import_graph_def(graph_def, name='')
                saver.restore(self.guided_sess, '/tmp/guided_backprop_ckpt')

                imported_y = self.guided_graph.get_tensor_by_name(y.name)
                imported_x = self.guided_graph.get_tensor_by_name(x.name)

                self.guided_grads_node = tf.gradients(
                    imported_y, imported_x)[0]

    def GetMask(self, x_value, feed_dict={}):
        """Returns a GuidedBackprop mask."""
        with self.guided_graph.as_default():
            guided_feed_dict = {}
            for tensor in feed_dict:
                guided_feed_dict[tensor.name] = feed_dict[tensor]
            guided_feed_dict[self.x.name] = [x_value]

        return self.guided_sess.run(
            self.guided_grads_node, feed_dict=guided_feed_dict)[0]


class IntegratedGradients(GradientSaliency):
    """A SaliencyMask class that implements the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """

    def GetMask(self, x_value, feed_dict={}, x_baseline=None, nsamples=100):
        """Returns a integrated gradients mask."""
        if x_baseline is None:
            x_baseline = np.zeros_like(x_value)

        assert x_baseline.shape == x_value.shape

        x_diff = x_value - x_baseline

        total_gradients = np.zeros_like(x_value)

        for alpha in np.linspace(0, 1, nsamples):
            x_step = x_baseline + alpha * x_diff

            total_gradients += super(IntegratedGradients, self).GetMask(
                x_step, feed_dict)

        return total_gradients * x_diff


class Occlusion(SaliencyMask):
    """A SaliencyMask class that computes saliency masks by occluding the image.
    This method slides a window over the image and computes how that occlusion
    affects the class score. When the class score decreases, this is positive
    evidence for the class, otherwise it is negative evidence.
    """

    def __init__(self, graph, session, y, x):
        super(Occlusion, self).__init__(graph, session, y, x)

    def GetMask(self, x_value, feed_dict={}, size=15, value=0):
        """Returns an occlusion mask."""
        occlusion_window = np.array([size, size, x_value.shape[2]])
        occlusion_window.fill(value)

        occlusion_scores = np.zeros_like(x_value)

        feed_dict[self.x] = [x_value]
        original_y_value = self.session.run(self.y, feed_dict=feed_dict)

        for row in range(x_value.shape[0] - size):
            for col in range(x_value.shape[1] - size):
                x_occluded = np.array(x_value)

                x_occluded[row:row + size, col:col +
                           size, :] = occlusion_window

                feed_dict[self.x] = [x_occluded]
                y_value = self.session.run(self.y, feed_dict=feed_dict)

                score_diff = original_y_value - y_value
                occlusion_scores[row:row + size,
                                 col:col + size, :] += score_diff
        return occlusion_scores
