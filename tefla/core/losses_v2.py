import six
import abc
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Loss(object):

    def __call__(self,
                 prediction_tensor,
                 target_tensor,
                 ignore_nan_targets=False,
                 scope=None,
                 **params):
        """Call the loss function.
        Args:
          prediction_tensor: a tensor representing predicted quantities.
          target_tensor: a tensor representing regression or classification targets.
          ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
          scope: Op scope name. Defaults to 'Loss' if None.
          **params: Additional keyword arguments for specific implementations of
                  the Loss.
        Returns:
          loss: a tensor representing the value of the loss function.
        """
        with tf.name_scope(scope, 'Loss',
                           [prediction_tensor, target_tensor, params]) as scope:
            if ignore_nan_targets:
                target_tensor = tf.where(tf.is_nan(target_tensor),
                                         prediction_tensor,
                                         target_tensor)
            return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abc.abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overriden by implementations.
        Args:
          prediction_tensor: a tensor representing predicted quantities
          target_tensor: a tensor representing regression or classification targets
          **params: Additional keyword arguments for specific implementations of
                  the Loss.
        Returns:
          loss: a tensor representing the value of the loss function
        """
        raise NotImplementedError


class WeightedLognLoss(Loss):
    """Log loss function with anchorwise output support.
    """

    def _compute_loss(self, prediction_tensor, target_tensor, eps=1e-7, weights=1.0):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
          eps: a constant to set upper or lower limit for labels, smoothening factor
          weights: a float tensor

        Returns:
          loss: a (scalar) tensor representing the value of the loss function
                or a float tensor of shape [batch_size, num_anchors]
        """
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        losses = -tf.multiply(labels, tf.log(predictions + eps)) - tf.multiply(
            (1 - labels), tf.log(1 - predictions + eps))
        return tf.losses.compute_weighted_loss(losses, weights)
