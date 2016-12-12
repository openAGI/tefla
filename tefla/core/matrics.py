# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


class Metric(object):
    def __init__(self, name=None):
        self.name = name

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
    def __init__(self, name='Top_K'):
        super(Top_k, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets, top_k=1):
        return self._top_k_op(predictions, targets, top_k=top_k)

    def _top_k_op(self, predictions, targets, top_k=1, dtype=tf.int32):
        with tf.name_scope('Top_' + str(top_k)):
            targets = tf.cast(targets, dtype)
            correct_pred = tf.nn.in_top_k(predictions, tf.argmax(targets, 1), top_k)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc


class Kappa(Metric, MetricMixin):
    def __init__(self, name='kappa'):
        super(Kappa, self).__init__(name)

    def metric(self, predictions, targets):
        targets = np.array(targets)
        predictions = np.array(predictions)
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets = targets.dot(range(targets.shape[1]))
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        try:
            return self._quadratic_weighted_kappa(predictions, targets)
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
        conf_mat = self.confusion_matrix(rater_a, rater_b, min_rating, max_rating)
        num_ratings = len(conf_mat)
        num_scored_items = float(len(rater_a))

        hist_rater_a = self.histogram(rater_a, min_rating, max_rating)
        hist_rater_b = self.histogram(rater_b, min_rating, max_rating)

        numerator = 0.0
        denominator = 0.0

        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items
        try:
            return 1.0 - numerator / denominator
        except ZeroDivisionError:
            return 0.0001


class KappaV2(Metric, MetricMixin):
    def __init__(self, name='kappa'):
        super(Kappa, self).__init__(name)

    def metric(self, predictions, targets, num_classes=5, batch_size=32, **kwargs):
        targets = np.array(targets)
        predictions = np.array(predictions)
        if targets.ndim == 1:
            targets = one_hot(targets, m=num_classes)
        if predictions.ndim == 1:
            predictions = one_hot(predictions, m=num_classes)
        return self._kappa_loss(predictions, targets, batch_size=batch_size, **kwargs)

    def _kappa_loss(self, predictions, labels, y_pow=1, eps=1e-15, num_ratings=5, batch_size=32, name='kappa'):
        with tf.name_scope(name):
            labels = tf.to_float(labels)
            predictions = tf.to_float(predictions)
            repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
            repeat_op_sq = tf.square((repeat_op -tf.transpose(repeat_op)))
            weights= repeat_op_sq / tf.to_float((num_ratings - 1) ** 2)

            pred_ = predictions ** y_pow
            try:
                pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
            except:
                pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [batch_size, 1]))

            hist_rater_a = tf.reduce_sum(pred_norm, 0)
            hist_rater_b = tf.reduce_sum(labels, 0)

            conf_mat = tf.matmul(tf.transpose(pred_norm), labels)
            print(pred_norm.get_shape())

            nom = tf.reduce_sum(weights * conf_mat)
            denom = tf.reduce_sum(weights * tf.matmul(tf.reshape(hist_rater_a, [num_ratings, 1]), tf.reshape(hist_rater_b, [1, num_ratings])) / tf.to_float(batch_size))

            try:
                return (1 - nom / denom)
            except:
                return (1 - nom / (denom + eps))


class Auroc(Metric):
    def __init__(self, name='auroc'):
        super(Auroc, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets):
        return self._auroc(predictions, targets)

    def _auroc(self, y_pred, y_true):
        try:
            return roc_auc_score(y_true, y_pred[:, 1])
        except ValueError as e:
            print e
            return accuracy_score(y_true, np.argmax(y_pred, axis=1))


class F1score(Metric):
    def __init__(self, name='auroc'):
        super(F1score, self).__init__(name)
        self.name = name

    def metric(self, predictions, targets):
        return self._f1_score(predictions, targets)

    def _f1_score(y_pred, y_true):
        y_pred_2 = np.argmax(y_pred, axis=1)
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred_2)
        return accuracy_score(y_true, y_pred_2) if 0 in f1 else np.mean(f1)


def accuracy_op(y_pred, y_true):
    if not isinstance(y_true, tf.Tensor):
        raise ValueError("mean_accuracy 'input' argument only accepts type "
                         "Tensor, '" + str(type(input)) + "' given.")
    with tf.name_scope('Accuracy'):
        acc = accuracy_score(y_true, y_pred.argmax(axis=1))
    return acc


def one_hot(vec, m=None):
    if m is None:
        m = int(np.max(vec)) + 1
    return np.eye(m)[vec].astype('int32')
