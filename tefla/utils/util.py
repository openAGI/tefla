import importlib
import logging
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables

from quadratic_weighted_kappa import quadratic_weighted_kappa


def roc(y_true, y_pred, classes=[0, 1, 2, 3, 4]):
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc


def float32(k):
    return np.cast['float32'](k)


def get_bbox_10crop(crop_size, im_size):
    im_center = im_size[:2] / 2.0
    h_indices = (0, (im_size[0] - crop_size[0]) / 2.0)
    w_indices = (0, (im_size[1] - crop_size[1]) / 2.0)
    bboxs = np.empty((5, 5), dtype=np.int32)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            bboxs[curr, :4] = (i, j, i + crop_size[0], j + crop_size[1])
            bboxs[curr, 4] = 1
            curr += 1
    bboxs[4, :4] = np.tile(im_center, (1, 2)) + np.concatenate([-crop_size / 2.0, crop_size / 2.0])
    bboxs[4, 4] = 1
    bboxs = np.tile(bboxs, (2, 1))
    bboxs[5:, 4] = 0

    return bboxs


def get_predictions(feature_net, images, tf=None, bbox=None, color_vec=None):
    tic = time.time()
    feature_net.batch_iterator_test.xform = tf
    feature_net.batch_iterator_test.crop_bbox = bbox
    feature_net.batch_iterator_test.standardizer.color_vec = color_vec
    predictions = feature_net.predict_proba(images)
    print('took %6.1f seconds' % (time.time() - tic))

    return predictions


def kappa_from_proba(w, p, y_true):
    return kappa(y_true, p.dot(w))


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def get_commit_sha():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                         stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.strip().decode('utf-8')


def get_submission_filename():
    sha = get_commit_sha()
    return "data/sub_{}_{}.csv".format(sha,
                                       datetime.now().replace(microsecond=0))


def f1_score_wrapper(y_true, y_pred):
    # y_pred_2 = np.asarray([1 if b > a else 0 for [a, b] in y_pred])
    y_pred_2 = np.argmax(y_pred, axis=1)
    # print("F1 score inputs:")
    # print(y_true)
    # print(y_pred_2)
    # print("---")
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred_2)
    return accuracy_score(y_true, y_pred_2) if 0 in f1 else np.mean(f1)


def auroc_wrapper(y_true, y_pred):
    # print("Auroc score inputs:")
    # print(y_true.tolist())
    # y_pred_2 = np.argmax(y_pred, axis=1)
    # print(y_pred_2)
    # print(y_pred[:, 1].tolist())
    # print("---")
    try:
        return roc_auc_score(y_true, y_pred[:, 1])
    except ValueError as e:
        print e
        return accuracy_score(y_true, np.argmax(y_pred, axis=1))


def kappa_wrapper(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = y_true.dot(range(y_true.shape[1]))
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        return quadratic_weighted_kappa(y_true, y_pred)
    except IndexError:
        return np.nan


kappa = kappa_wrapper


def accuracy_wrapper(y_true, y_pred):
    return accuracy_score(y_true, y_pred.argmax(axis=1))


def dump_vars(sess):
    all_vars = set(tf.all_variables())
    trainable_vars = set(tf.trainable_variables())
    non_trainable_vars = all_vars.difference(trainable_vars)

    def _dump_set(var_set):
        names_vars = map(lambda v: (v.name, v), var_set)
        for n, v in sorted(names_vars, key=lambda nv: nv[0]):
            print("%s=%s" % (n, sess.run(v)))

    print("Variable values:")
    print("-----------")
    print("\n---Trainable vars:")
    _dump_set(trainable_vars)
    print("\n---Non Trainable vars:")
    _dump_set(non_trainable_vars)
    print("-----------")


def init_logging(file_name, file_log_level, console_log_level, clean=False):
    import sys
    logger = logging.getLogger('tefla')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    h1 = logging.StreamHandler(stream=sys.stdout)
    h1.setLevel(console_log_level)
    h1.setFormatter(formatter)
    logger.addHandler(h1)

    mode = 'w' if clean else 'a'
    h2 = logging.FileHandler(file_name, mode=mode)
    h2.setLevel(file_log_level)
    h2.setFormatter(formatter)
    logger.addHandler(h2)


def get_input_shape(x):
    "Return input shape"
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    elif type(x) in [np.ndarray, list, tuple]:
        return np.shape(x)
    else:
        raise Exception("Invalid input layer")


def valid_dtypes():
    """Valid types for loss, variables and gradients.
        Subclasses should override to allow other float types.
    Returns:
        Valid types for loss, variables and gradients.
    """
    return set([dtypes.float16, dtypes.float32, dtypes.float64])


def assert_valid_dtypes(tensors):
    """Asserts tensors are all valid types (see `_valid_dtypes`).
    Args:
        tensors: Tensors to check.
    Raises:
        ValueError: If any tensor is not a valid type.
    """
    valid_dtype = valid_dtypes()
    for t in tensors:
        dtype = t.dtype.base_dtype
        if dtype not in valid_dtype:
            raise ValueError("Invalid type %r for %s, expected: %s." % (dtype, t.name, [v for v in valid_dtype]))


def constant_value(value_or_tensor_or_var, dtype=None):
    """Returns value if value_or_tensor_or_var has a constant value.

    Args:
        value_or_tensor_or_var: A value, a `Tensor` or a `Variable`.
        dtype: Optional `tf.dtype`, if set it would check it has the right
          dtype.

    Returns:
        The constant value or None if it not constant.

    Raises:
        ValueError: if value_or_tensor_or_var is None or the tensor_variable has the
                    wrong dtype.
    """
    if value_or_tensor_or_var is None:
        raise ValueError('value_or_tensor_or_var cannot be None')
    value = value_or_tensor_or_var
    if isinstance(value_or_tensor_or_var, (ops.Tensor, variables.Variable)):
        if dtype and value_or_tensor_or_var.dtype != dtype:
            raise ValueError('It has the wrong type %s instead of %s' % (value_or_tensor_or_var.dtype, dtype))
        if isinstance(value_or_tensor_or_var, variables.Variable):
            value = None
        else:
            value = tensor_util.constant_value(value_or_tensor_or_var)
    return value


def static_cond(pred, fn1, fn2):
    """Return either fn1() or fn2() based on the boolean value of `pred`.

    Same signature as `control_flow_ops.cond()` but requires pred to be a bool.

    Args:
        pred: A value determining whether to return the result of `fn1` or `fn2`.
        fn1: The callable to be performed if pred is true.
        fn2: The callable to be performed if pred is false.

    Returns:
        Tensors returned by the call to either `fn1` or `fn2`.

    Raises:
        TypeError: if `fn1` or `fn2` is not callable.
    """
    if not callable(fn1):
        raise TypeError('fn1 must be callable.')
    if not callable(fn2):
        raise TypeError('fn2 must be callable.')
    if pred:
        return fn1()
    else:
        return fn2()


def smart_cond(pred, fn1, fn2, name=None):
    """Return either fn1() or fn2() based on the boolean predicate/value `pred`.

    If `pred` is bool or has a constant value it would use `static_cond`,
     otherwise it would use `tf.cond`.

    Args:
        pred: A scalar determining whether to return the result of `fn1` or `fn2`.
        fn1: The callable to be performed if pred is true.
        fn2: The callable to be performed if pred is false.
        name: Optional name prefix when using tf.cond
    Returns:
        Tensors returned by the call to either `fn1` or `fn2`.
    """
    pred_value = constant_value(pred)
    if pred_value is not None:
        # Use static_cond if pred has a constant value.
        return static_cond(pred_value, fn1, fn2)
    else:
        # Use dynamic cond otherwise.
        return control_flow_ops.cond(pred, fn1, fn2, name)


def get_variable_collections(variables_collections, name):
    if isinstance(variables_collections, dict):
        variable_collections = variables_collections.get(name, None)
    else:
        variable_collections = variables_collections
    return variable_collections


def veryify_args(actual, allowed_keys, msg_prefix):
    actual_keys = set(actual.keys())
    extra = list(actual_keys - set(allowed_keys))
    if len(extra) > 0:
        raise ValueError("%s %s" % (msg_prefix, extra))
