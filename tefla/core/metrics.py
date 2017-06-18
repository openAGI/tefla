# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import abc
import six
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from ..convert import convert
from ..convert_labels import convert_labels


@six.add_metaclass(abc.ABCMeta)
class Metric(object):

    def __init__(self, name=None):
        self.name = name

    @abc.abstractmethod
    def metric(self, predictions, targets, **kwargs):
        raise NotImplementedError


class MetricMixin(object):

    def __init__(self, name='metric_hist_confusion'):
        self.name = name
        super(MetricMixin, self).__init__(name)

    def confusion_matrix(self, rater_a, rater_b, min_rating=None, max_rating=None):
        """
        Returns the confusion matrix between rater's ratings
        """
        assert(len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(rater_a + rater_b)
        if max_rating is None:
            max_rating = max(rater_a + rater_b)
        num_ratings = int(max_rating - min_rating + 1)
        conf_mat = [[0 for i in range(num_ratings)]
                    for j in range(num_ratings)]
        for a, b in zip(rater_a, rater_b):
            conf_mat[a - min_rating][b - min_rating] += 1
        return conf_mat

    def histogram(self, ratings, min_rating=None, max_rating=None):
        """
        Returns the counts of each type of rating that a rater made
        """
        if min_rating is None:
            min_rating = min(ratings)
        if max_rating is None:
            max_rating = max(ratings)
        num_ratings = int(max_rating - min_rating + 1)
        hist_ratings = [0 for x in range(num_ratings)]
        for r in ratings:
            hist_ratings[r - min_rating] += 1
        return hist_ratings


class Top_k(Metric):
    """
    Class to compute Top_k accuracy metric for predictions and labels
    """

    def __init__(self, name='Top_K'):
        super(Top_k, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets, top_k=1):
        """
        Computes top k metric

        Args:
            predictions: 2D tensor/array, predictions of the network
            targets: 2D tensor/array, ground truth labels of the network
            top_k: int, returns the top_k accuracy; {1,2,3 max_classes}

        Returns:
            top_k accuracy
        """
        return self._top_k_op(predictions, targets, top_k=top_k)

    def _top_k_op(self, predictions, targets, top_k=1, dtype=tf.int32):
        with tf.name_scope('Top_' + str(top_k)):
            targets = tf.cast(targets, dtype)
            correct_pred = tf.nn.in_top_k(
                predictions, tf.argmax(targets, 1), top_k)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc


class IOU(Metric, MetricMixin):
    """
    Class to compute IOU metric for predictions and labels
    """

    def __init__(self, name='IOU'):
        super(IOU, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets, top_k=1):
        """
        Computes top k metric

        Args:
            predictions: 2D tensor/array, predictions of the network
            targets: 2D tensor/array, ground truth labels of the network
            top_k: int, returns the top_k accuracy; {1,2,3 max_classes}

        Returns:
            top_k accuracy
        """
        return self._iou_op(predictions, targets)

    def _iou_op(self, predictions, targets, top_k=1, dtype=tf.int32):
        with tf.name_scope('iou'):
            targets = tf.cast(targets, dtype)
            conf_mat = self.confusion_matrix(predictions, targets)
            t = []
            k = len(conf_mat[0])
            for i in range(k):
                t.append(sum([conf_mat[i][j] for j in range(k)]))
            return (1.0 / k) * sum([float(conf_mat[i][i]) / (t[i] - conf_mat[i][i] + sum([conf_mat[j][i] for j in range(k)])) for i in range(k)])


class IOUSeg(object):

    def __init__(self, name='IOU'):
        self.name = name
        super(IOUSeg, self).__init__()

    def meaniou(self, predictor, predict_dir, image_size):
        segparams = util.SegParams()
        classes = segparams.feature_classes().values()
        num_classes = len(classes) + 1
        hist = np.zeros((num_classes, num_classes))
        image_names = [filename.strip() for filename in os.listdir(
            predict_dir) if filename.endswith('.jpg')]
        for image_filename in image_names:
            final_prediction_map = predictor.predict(
                os.path.join(predict_dir, image_filename))
            final_prediction_map = final_prediction_map.transpose(
                0, 2, 1).squeeze()
            gt_name = os.path.join(predict_dir,
                                   image_filename[:-4] + '_final_mask' + '.png')
            gt = convert(gt_name, image_size)
            gt = np.asarray(gt)
            gt = convert_labels(gt, image_size, image_size)
            hist += compute_hist(gt, final_prediction_map,
                                 num_classes=num_classes)
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        meaniou = np.nanmean(iou)

        return meaniou

    def per_class_iou(self, predictor, predict_dir, image_size):
        image_names = [filename.strip() for filename in os.listdir(
            predict_dir) if filename.endswith('.jpg')]
        per_class_iou_hist = defaultdict(np.ndarray)
        per_class_iou_dict = defaultdict(float)
        segparams = util.SegParams()
        classes = segparams.feature_classes().values()
        num_classes = len(classes) + 1
        for class_id, class_name in enumerate(classes, 1):
            per_class_iou_hist[class_name] = np.zeros(
                (num_classes, num_classes))

        for image_filename in image_names:
            final_prediction_map = predictor.predict(
                os.path.join(predict_dir, image_filename))
            final_prediction_map = final_prediction_map.transpose(
                0, 2, 1).squeeze()
            gt_name = os.path.join(predict_dir,
                                   image_filename[:-4] + '_final_mask' + '.png')
            gt = convert(gt_name, image_size)
            gt = np.asarray(gt)
            gt = convert_labels(gt, image_size, image_size)
            for class_id, class_name in enumerate(classes, 1):
                per_class_iou_hist[class_name] += compute_hist(np.asarray(gt == class_id, dtype=np.int32), np.asarray(
                    final_prediction_map == class_id, dtype=np.int32), num_classes=num_classes)

        for class_id, class_name in enumerate(classes, 1):
            per_class_iou_dict[class_name] = np.nanmean(np.diag(per_class_iou_hist[class_name]) / (per_class_iou_hist[
                                                        class_name].sum(1) + per_class_iou_hist[class_name].sum(0) - np.diag(per_class_iou_hist[class_name])))
        return per_class_iou_dict


class Kappa(Metric, MetricMixin):

    def __init__(self, name='kappa'):
        super(Kappa, self).__init__(name)

    def metric(self, predictions, targets, num_classes):
        """
        Computes Kappa metric

        Args:
            predictions: 2D tensor/array, predictions of the network
            targets: 2D tensor/array, ground truth labels of the network

        Returns:
            Kappa score
        """
        targets = np.array(targets)
        predictions = np.array(predictions)
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets = targets.dot(range(targets.shape[1]))
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        try:
            return self._quadratic_weighted_kappa(predictions, targets, max_rating=num_classes - 1)
        except IndexError:
            return np.nan

    def _quadratic_weighted_kappa(self, rater_a, rater_b, min_rating=0, max_rating=4):
        """
        Calculates the quadratic weighted kappa
        quadratic_weighted_kappa calculates the quadratic weighted kappa
        value, which is a measure of inter-rater agreement between two raters
        that provide discrete numeric ratings.  Potential values range from -1
        (representing complete disagreement) to 1 (representing complete
        agreement).  A kappa value of 0 is expected if all agreement is due to
        chance.

        quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
        each correspond to a list of integer ratings.  These lists must have the
        same length.

        The ratings should be integers, and it is assumed that they contain
        the complete range of possible ratings.

        quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
        is the minimum possible rating, and max_rating is the maximum possible
        rating
        """
        rater_a = np.clip(rater_a, min_rating, max_rating)
        rater_b = np.clip(rater_b, min_rating, max_rating)

        rater_a = np.round(rater_a).astype(int).ravel()
        rater_a[~np.isfinite(rater_a)] = 0
        rater_b = np.round(rater_b).astype(int).ravel()
        rater_b[~np.isfinite(rater_b)] = 0

        assert(len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(min(rater_a), min(rater_b))
        if max_rating is None:
            max_rating = max(max(rater_a), max(rater_b))
        conf_mat = self.confusion_matrix(
            rater_a, rater_b, min_rating, max_rating)
        num_ratings = len(conf_mat)
        num_scored_items = float(len(rater_a))

        hist_rater_a = self.histogram(rater_a, min_rating, max_rating)
        hist_rater_b = self.histogram(rater_b, min_rating, max_rating)

        numerator = 0.0
        denominator = 0.0

        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (
                    hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items
        try:
            return 1.0 - numerator / denominator
        except ZeroDivisionError:
            return 0.0001


class KappaV2(Metric, MetricMixin):

    def __init__(self, name='kappa', num_classes=10, batch_size=32):
        self.num_classes = num_classes
        self.batch_size = batch_size
        super(KappaV2, self).__init__(name)

    def metric(self, predictions, targets, num_classes=None, batch_size=None, **kwargs):
        """
        Computes Kappa metric

        Args:
            predictions: 2D tensor/array, predictions of the network
            targets: 2D tensor/array, ground truth labels of the network
            num_classes: int, num_classes of the network
            batch_size: batch_size for predictions of the network

        Returns:
            Kappa score
        """
        if num_classes is None:
            num_classes = self.num_classes
        if batch_size is None:
            batch_size = self.batch_size
        targets = tf.convert_to_tensor(targets)
        predictions = tf.convert_to_tensor(predictions)
        if targets.get_shape().ndims == 1:
            targets = tf.one_hot(targets, num_classes, on_value=1, off_value=0)
        if predictions.get_shape().ndims == 1:
            predictions = tf.one_hot(
                predictions, num_classes, on_value=1, off_value=0)
        return self._kappa_loss(predictions, targets, batch_size=batch_size, num_ratings=num_classes, **kwargs)

    def _kappa_loss(self, predictions, labels, y_pow=1, eps=1e-15, num_ratings=5, batch_size=32, name='kappa'):
        with tf.name_scope(name):
            labels = tf.to_float(labels)
            predictions = tf.to_float(predictions)
            repeat_op = tf.to_float(tf.tile(tf.reshape(
                tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
            repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
            weights = repeat_op_sq / tf.to_float((num_ratings - 1) ** 2)

            pred_ = predictions ** y_pow
            try:
                pred_norm = pred_ / \
                    (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
            except Exception as e:
                print(e.message)
                pred_norm = pred_ / \
                    (eps + tf.reshape(tf.reduce_sum(pred_, 1),
                                      [batch_size, 1]))

            hist_rater_a = tf.reduce_sum(pred_norm, 0)
            hist_rater_b = tf.reduce_sum(labels, 0)

            conf_mat = tf.matmul(tf.transpose(pred_norm), labels)

            nom = tf.reduce_sum(weights * conf_mat)
            denom = tf.reduce_sum(weights * tf.matmul(tf.reshape(hist_rater_a, [
                                  num_ratings, 1]), tf.reshape(hist_rater_b, [1, num_ratings])) / tf.to_float(batch_size))

            try:
                return (1 - nom / denom)
            except Exception as e:
                print(e.message)
                return (1 - nom / (denom + eps))


class Auroc(Metric):

    def __init__(self, name='auroc'):
        super(Auroc, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets, num_classes=5):
        """
        Computes auroc metric

        Args:
            predictions: 2D tensor/array, predictions of the network
            targets: 2D tensor/array, ground truth labels of the network
            num_classes: int, num_classes of the network

        Returns:
            auroc score
        """
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        if predictions.ndim == 1:
            predictions = one_hot(predictions, m=num_classes)
        return self._auroc(predictions, targets)

    def _auroc(self, y_pred, y_true):
        try:
            return roc_auc_score(y_true, y_pred[:, 1])
        except ValueError as e:
            print(e)
            return accuracy_score(y_true, np.argmax(y_pred, axis=1))


class F1score(Metric):

    def __init__(self, name='auroc'):
        super(F1score, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets, num_classes=5):
        """
        Computes F1 metric

        Args:
            predictions: 2D tensor/array, predictions of the network
            targets: 2D tensor/array, ground truth labels of the network
            num_classes: int, num_classes of the network

        Returns:
            F1 score
        """
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        if predictions.ndim == 1:
            predictions = one_hot(predictions, m=num_classes)
        return self._f1_score(predictions, targets)

    def _f1_score(self, y_pred, y_true):
        y_pred_2 = np.argmax(y_pred, axis=1)
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred_2)
        return accuracy_score(y_true, y_pred_2) if 0 in f1 else np.mean(f1)


def accuracy_op(predictions, targets, num_classes=5):
    """
    Computes accuracy metric

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network
        num_classes: int, num_classes of the network

    Returns:
        accuracy
    """
    with tf.name_scope('Accuracy'):
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        if predictions.ndim == 1:
            predictions = one_hot(predictions, m=num_classes)
        acc = accuracy_score(targets, np.argmax(predictions, axis=1))
    return acc


def one_hot(vec, m=None):
    """
    Retruns one hot vector

    Args:
        vec: a vector
        m: num_classes
    """
    if m is None:
        m = int(np.max(vec)) + 1
    return np.eye(m)[vec].astype('int32')


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(preds, gt, num_classes=15):
    hist = np.zeros((num_classes, num_classes))
    hist += fast_hist(np.reshape(gt, (-1)),
                      np.reshape(preds, (-1)), num_classes)
    return hist


def dice_coef(y_true, y_pred):
    """ Compute dice coef

    Args:
        y_true: a 2-D `array`, ground truth label
        y_pred: q 2-D `array`, prediction

    Returns:
        a `float`, dice value
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (np.sum(y_true_f) + np.sum(y_pred_f) + 100)


def char_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes character level accuracy.
    Both predictions and targets should have the same shape
    [batch_size x seq_length].

    Args:
        predictions: predicted characters ids.
        targets: ground truth character ids.
        rej_char: the character id used to mark an empty element (end of sequence).
        streaming: if True, uses the streaming mean from the slim.metric module.

    Returns:
        a update_ops for execution and value tensor whose value on evaluation
            returns the total character accuracy.
    """
    with tf.variable_scope('CharAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.to_int32(targets)
        const_rej_char = tf.constant(rej_char, shape=targets.get_shape())
        weights = tf.to_float(tf.not_equal(targets, const_rej_char))
        correct_chars = tf.to_float(tf.equal(predictions, targets))
        accuracy_per_example = tf.div(tf.reduce_sum(tf.multiply(
            correct_chars, weights), 1),  tf.reduce_sum(weights, 1))
        if streaming:
            return tf.contrib.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)


def sequence_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes sequence level accuracy.
    Both input tensors should have the same shape: [batch_size x seq_length].

    Args:
        predictions: predicted character classes.
        targets: ground truth character classes.
        rej_char: the character id used to mark empty element (end of sequence).
        streaming: if True, uses the streaming mean from the slim.metric module.

    Returns:
        a update_ops for execution and value tensor whose value on evaluation
            returns the total sequence accuracy.
    """

    with tf.variable_scope('SequenceAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.to_int32(targets)
        const_rej_char = tf.constant(
            rej_char, shape=targets.get_shape(), dtype=tf.int32)
        include_mask = tf.not_equal(targets, const_rej_char)
        include_predictions = tf.to_int32(
            tf.where(include_mask, predictions, tf.zeros_like(predictions) + rej_char))
        correct_chars = tf.to_float(tf.equal(include_predictions, targets))
        correct_chars_counts = tf.cast(
            tf.reduce_sum(correct_chars, reduction_indices=[1]), dtype=tf.int32)
        target_length = targets.get_shape().dims[1].value
        target_chars_counts = tf.constant(
            target_length, shape=correct_chars_counts.get_shape())
        accuracy_per_example = tf.to_float(
            tf.equal(correct_chars_counts, target_chars_counts))
        if streaming:
            return tf.contrib.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)
